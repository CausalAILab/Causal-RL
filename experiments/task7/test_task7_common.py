"""Checks for experiments/task7/common.py (python -m unittest experiments/task7/test_task7_common.py)."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

from causal_rl.algo.imitation.gail.core_net import ContinuousActor  # noqa: E402
from causal_rl.algo.imitation.imitate import ContinuousPolicyNN, make_window_spec  # noqa: E402
from causal_rl.algo.imitation.sqil.causal_sqil import SQILReplayBuffer, initialize_expert_buffer  # noqa: E402

SHORT = 60  # short episodes keep the tests fast


def random_policy(name, seed):
    rng = np.random.default_rng(seed)
    A = C.spec(name)["dims"]["X"]
    return lambda obs, t: rng.uniform(-1, 1, A).astype(np.float32)


def live_episode(name, encoder, seed):
    """Run a short train-env episode and record encode(live history, t) at every step."""
    env = C.make_env(name, "train", seed=seed, num_steps=SHORT)
    pol = random_policy(name, seed)
    obs, _ = env.reset(seed=seed)
    zs = []
    for t in range(SHORT):
        zs.append(encoder(obs, t))
        a = pol(obs, t)
        obs, _, term, trunc, _ = env.do(lambda _o, a=a: a)
        if term or trunc:
            break
    return zs, env


class Inputs(unittest.TestCase):
    def test_dimensions(self):
        expect = {"antmed": (355, 377), "hummed": (240, 246)}
        for name, (dc, dv) in expect.items():
            with self.subTest(env=name):
                c, v, w = (C.Encoder(name, e) for e in C.ENCODINGS)
                self.assertEqual((c.dim, v.dim, w.dim), (dc, dv, dv))
                self.assertEqual({s for s in v.slots} - {s for s in c.slots},
                                 {("W", lag) for lag in range(-C.spec(name)["k"], 1)})

    def test_wmask_zeroes_only_w(self):
        enc_v, enc_w = C.Encoder("antmed", "vanilla"), C.Encoder("antmed", "wmask")
        zs_v, _ = live_episode("antmed", enc_v, 3)
        zs_w, _ = live_episode("antmed", enc_w, 3)
        keep = enc_w.mask.astype(bool)
        for zv, zw in zip(zs_v, zs_w):
            np.testing.assert_array_equal(zv[keep], zw[keep])
            self.assertTrue(np.all(zw[~keep] == 0))

    def test_stored_episode_encodes_like_live_history(self):
        """Encoding the compact stored arrays equals encoding the live per-step history."""
        for name in ("antmed", "hummed"):
            for encoding in ("causal", "vanilla"):
                with self.subTest(env=name, encoding=encoding):
                    enc = C.Encoder(name, encoding)
                    zs, env = live_episode(name, enc, 11)
                    scm = env.env
                    stored = {v: np.asarray(getattr(scm, v), np.float32) for v in C.spec(name)["state_vars"]}
                    stored.update(X=np.asarray(scm.X, np.float32), reward=np.asarray(scm._Y, np.float32),
                                  U=np.asarray(scm._U, np.float32), terminated=False)
                    with tempfile.TemporaryDirectory() as d:
                        p = Path(d) / "demo.npz"
                        C.save_episodes(p, name, [stored], {"test": True})
                        demos = C.Demos(p, name)
                        obs = demos.obs(0)
                        self.assertNotIn(C.spec(name)["hidden"], obs)
                        self.assertNotIn("U", obs)
                        self.assertNotIn("reward", obs)
                        for t, z in enumerate(zs):
                            np.testing.assert_allclose(enc(obs, t), z, rtol=0, atol=1e-6)

    def test_records_fill_expert_buffer(self):
        name = "antmed"
        env = C.make_env(name, "expert", seed=0, num_steps=SHORT)
        episodes = [C.run_episode(env, random_policy(name, s), s, SHORT) for s in (5, 6)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "demo.npz"
            C.save_episodes(p, name, episodes, {})
            demos = C.Demos(p, name)
            recs = demos.records(range(len(demos)))
            self.assertEqual(len(recs), demos.n_transitions(range(2)))
            buf = SQILReplayBuffer(10_000, 0.5)
            enc = C.Encoder(name, "causal")
            initialize_expert_buffer(recs, enc, buf, torch.device("cpu"))
            self.assertEqual(len(buf.expert_buffer), len(recs))
            last = [r for r in recs if r["step"] == demos.lengths[r["episode"]] - 1]
            self.assertTrue(all(r["truncated"] or r["terminated"] for r in last))


class Evaluation(unittest.TestCase):
    def test_isolated_and_repeatable(self):
        name = "antmed"
        enc = C.Encoder(name, "causal")
        torch.manual_seed(0)
        actor = ContinuousActor(num_inputs=enc.dim, num_outputs=8, hidden_size=64, std=0.0, num_blocks=1,
                                dropout=0.05, layernorm=True)
        actor.train()
        np.random.seed(123)
        torch_state, np_state = torch.get_rng_state(), np.random.get_state()
        seeds = [C.seed_of("final_eval", 0, 1000, i) for i in range(2)]
        orig_H = C.SPECS[name]["H"]
        C.SPECS[name]["H"] = SHORT
        try:
            r1 = C.evaluate(name, C.actor_policy(actor, enc), seeds, c=1.0, modules=[actor])
            r2 = C.evaluate(name, C.actor_policy(actor, enc), seeds, c=1.0, modules=[actor])
        finally:
            C.SPECS[name]["H"] = orig_H
        self.assertEqual(r1, r2)
        self.assertTrue(actor.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(np.random.get_state()[1].tolist(), np_state[1].tolist())

    def test_c_changes_only_w(self):
        """Same seed and actions at c=0 and c=1: identical physics and rewards, only W differs."""
        name = "antmed"
        acts = [np.random.default_rng(4).uniform(-1, 1, 8).astype(np.float32) for _ in range(40)]
        out = {}
        for c in (0.0, 1.0):
            env = C.make_env(name, "eval", seed=0, c=c, num_steps=SHORT)
            out[c] = C.run_episode(env, lambda o, t: acts[t], 77, 40)
        for v in ("P", "A", "L", "T", "J", "reward", "U"):
            np.testing.assert_array_equal(out[0.0][v], out[1.0][v])
        self.assertFalse(np.allclose(out[0.0]["W"], out[1.0]["W"]))


class Expert(unittest.TestCase):
    def test_load_scil_style_checkpoint_and_collect(self):
        name = "antmed"
        s = C.spec(name)
        lookback = 5
        env = C.make_env(name, "expert", seed=0, num_steps=SHORT)
        obs_prefix = env.env.observed_unobserved_vars[0]
        dims = {v: d for v, d in s["dims"].items() if v != "W"}
        slots = make_window_spec(include_vars=[v for v in obs_prefix if v in dims], dims=dims, lookback=lookback)
        input_dim = sum(d for _, _, d in slots) + len(slots)
        model = ContinuousPolicyNN(input_dim=input_dim, action_dim=8, hidden_dim=32, num_blocks=1, dropout=0.0,
                                   layernorm=True, final_tanh=True,
                                   action_bounds=(env.env.action_space.low, env.env.action_space.high))
        Z_trim = {f"X{t}": {f"{v}{j}" for v in dims for j in range(max(0, t - lookback), t + (v != "X"))}
                  for t in range(SHORT)}
        ck = {"state_dict": model.state_dict(), "slots": slots, "Z_trim": Z_trim, "input_dim": input_dim,
              "num_actions": 8, "hidden_dim": 32, "num_blocks": 1, "dropout": 0.0, "layernorm": True,
              "final_tanh": True, "action_bounds_low": env.env.action_space.low,
              "action_bounds_high": env.env.action_space.high, "lookback": lookback}
        with tempfile.TemporaryDirectory() as d:
            torch.save(ck, Path(d) / "expert.pt")
            pi, _ = C.load_expert(Path(d) / "expert.pt")
            ep1 = C.run_episode(env, lambda o, t: pi(o), 9, SHORT)
            ep2 = C.run_episode(env, lambda o, t: pi(o), 9, SHORT)
        np.testing.assert_array_equal(ep1["X"], ep2["X"])  # deterministic expert + seeded reset
        self.assertEqual(len(ep1["X"]), SHORT)
        self.assertIn("O", ep1)


class Seeds(unittest.TestCase):
    def test_streams_disjoint(self):
        a = {C.seed_of("final_eval", e, s, i) for e in range(4) for s in range(1000, 1010) for i in range(1000)}
        b = {C.seed_of("checkpoint_eval", e, s, i) for e in range(4) for s in range(1000, 1010) for i in range(1000)}
        d = {C.seed_of("demo", e, 0, i) for e in range(4) for i in range(1000)}
        self.assertFalse(a & b or a & d or b & d)


if __name__ == "__main__":
    unittest.main()
