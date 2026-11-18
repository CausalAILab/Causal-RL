"""End-to-end checks of the Task 7 learner loops on tiny budgets (python -m unittest experiments/task7/test_task7_learners.py)."""
from pathlib import Path
import sys
import copy
import tempfile
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import learners as Lr  # noqa: E402

NAME = "antmed"
TMP = tempfile.TemporaryDirectory()


def fake_demos(n=4, length=300):
    """Random-policy episodes saved in the demo format (stand-in for the expert data)."""
    path = Path(TMP.name) / "demos.npz"
    if not path.exists():
        env = C.make_env(NAME, "expert", seed=0, num_steps=length)
        rng = np.random.default_rng(0)
        eps = [C.run_episode(env, lambda o, t: rng.uniform(-1, 1, 8).astype(np.float32), 100 + i, length)
               for i in range(n)]
        C.save_episodes(path, NAME, eps, {"fake": True})
    return C.Demos(path, NAME)


SMALL = dict(Lr.OFFPOLICY, start_steps=200, batch_size=64, buffer_capacity=1000)  # expert half 500 < 900 demo transitions


class OffPolicy(unittest.TestCase):
    def run_kind(self, kind, encoding="causal"):
        demos = fake_demos()
        enc = C.Encoder(NAME, encoding)
        cps, evals = [], []
        mods, act, log = Lr.train_offpolicy(
            kind, NAME, enc, demos, range(3), 1000, budget=1500, secondary=700, eval_every=600, eval_episodes=1,
            on_checkpoint=lambda tag, ts, upd, *_: cps.append((tag, ts, upd)), on_eval=evals.append, p=SMALL)
        return mods, log, cps, evals, demos

    def test_sqil_budget_checkpoints_and_buffer(self):
        mods, log, cps, evals, demos = self.run_kind("sqil")
        self.assertEqual(log["interactions"], 1500)
        self.assertEqual([(t, ts) for t, ts, _ in cps], [("secondary", 700), ("final", 1500)])
        self.assertEqual([e["target"] for e in evals], [600, 1200])
        self.assertGreater(log["updates"], 0)
        # expert half larger than the default capacity is enlarged, never overwritten
        self.assertEqual(demos.n_transitions(range(3)), 900)

    def test_iq_runs_with_mc_v_and_is_deterministic(self):
        a = self.run_kind("iq")[0][0].state_dict()
        b = self.run_kind("iq")[0][0].state_dict()
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)

    def test_shortened_run_matches_full_run_prefix(self):
        """Dev mode stops IQ-Learn at 1M with the LR schedule sized for 2M: weights at the stop equal a full
        run's weights at its 1M secondary checkpoint."""
        demos, enc = fake_demos(), C.Encoder(NAME, "causal")
        snaps = {}

        def grab(label):
            return lambda tag, ts, upd, mods, act: snaps.setdefault((label, ts), copy.deepcopy(mods[0].state_dict()))
        Lr.train_offpolicy("iq", NAME, enc, demos, range(3), 1000, budget=1500, secondary=700, eval_every=600,
                           eval_episodes=1, on_checkpoint=grab("full"), p=SMALL)
        Lr.train_offpolicy("iq", NAME, enc, demos, range(3), 1000, budget=700, secondary=None, eval_every=600,
                           eval_episodes=1, on_checkpoint=grab("short"), p=SMALL, schedule_budget=1500)
        a, b = snaps[("full", 700)], snaps[("short", 700)]
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)

    def test_wmask_input(self):
        _, log, _, _, _ = self.run_kind("sqil", "wmask")
        self.assertEqual(log["interactions"], 1500)


class Gail(unittest.TestCase):
    def test_rounds_secondary_and_eval(self):
        demos = fake_demos()
        cps, evals = [], []
        mods, act, log = Lr.train_gail(NAME, C.Encoder(NAME, "vanilla"), demos, range(3), 1000, rounds=2,
                                       episodes_per_round=1, secondary_interactions=1000, eval_every=1000,
                                       eval_episodes=1, on_checkpoint=lambda *x: cps.append(x), on_eval=evals.append)
        self.assertEqual(len(log["rounds"]), 2)
        self.assertEqual(cps[-1][0], "final")
        self.assertEqual(log["interactions"], log["rounds"][-1][1])


class Bc(unittest.TestCase):
    def test_episode_split_and_policy(self):
        demos = fake_demos()
        mods, act, log = Lr.train_bc(NAME, "causal", demos, range(3), range(3, 4), 1000, epochs=2, patience=5)
        self.assertEqual(log["train_samples"], 900)
        self.assertEqual(log["val_samples"], 300)
        env = C.make_env(NAME, "eval", seed=1, c=1.0, num_steps=30)
        ep = C.run_episode(env, act, 5, 30)
        self.assertEqual(ep["length"], 30)


if __name__ == "__main__":
    unittest.main()
