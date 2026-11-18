"""Task 6 protocol, environment-contract and runner checks (python test_task6.py)."""
import copy
import itertools
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as train

train.source_paths()


class Task6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = train.configuration(HERE / "config.json")
        cls.smoke = train.configuration(HERE / "config.json", "smoke")

    def test_main_design(self):
        cfg = self.main
        self.assertEqual(cfg["seeds"], list(range(1000, 1010)))
        envs = [e["id"] for _, e in train.enabled(cfg)]
        self.assertEqual(envs, ["robotwalk", "mdp_balanced", "drone_u2_h4", "drone_b2_h4",
                                "drone_r4_h8", "drone_r6_h10", "cartpole_calm"])
        self.assertNotIn("lava_s5", envs)
        kinds = {k: sum(1 for j in train.jobs(cfg) if j["kind"] == k) for k in ("neural", "reference", "bandit")}
        self.assertEqual(kinds, {"neural": 7 * 3 * 2 * 10 + 3 * 2 * 10, "reference": 7 * 2 * 10, "bandit": 20})
        self.assertEqual(cfg["schedules"]["short"]["checkpoints"], [0, 10240, 25600, 51200, 102400])
        long = cfg["schedules"]["long"]["checkpoints"]
        self.assertEqual((long[0], long[-1], len(long)), (0, 1048576, 33))
        keys = [j["key"] for j in train.jobs(cfg)]
        self.assertEqual(len(keys), len(set(keys)))
        # Most expensive jobs are launched first.
        self.assertEqual(train.jobs(cfg)[0]["algorithm"], "c51")

    def test_evaluation_seed_streams_are_disjoint(self):
        cfg = self.main
        seen = set()
        for (index, _), seed in itertools.product(train.enabled(cfg), cfg["seeds"]):
            for stream, n in (("checkpoint", cfg["evaluation"]["checkpoint_episodes"]),
                              ("final", cfg["evaluation"]["final_episodes"])):
                batch = {train.eval_seed(index, seed, stream, e) for e in range(n)}
                self.assertFalse(batch & seen)
                seen |= batch

    def test_inputs_public_bounded_and_paired(self):
        for _, spec in train.enabled(self.main):
            do, ctf = train.make_env(spec, "do"), train.make_env(spec, "ctf")
            n = do.action_space.n
            self.assertEqual(ctf.observation_space.shape[0], do.observation_space.shape[0] + n)
            od, info_d = do.reset(seed=5)
            oc, info_c = ctf.reset(seed=5)
            self.assertIsNone(info_d.get("next_natural_action"))
            self.assertIsNotNone(info_c.get("next_natural_action"))
            np.testing.assert_array_equal(od[:-1], oc[:len(od) - 1])
            self.assertEqual(oc[len(od) - 1:-1].sum(), 1.0)
            for o in (od, oc):
                self.assertTrue(np.all((o >= 0) & (o <= 1)))
            do.close(); ctf.close()

    def test_gym_make_every_environment(self):
        """CleanRL builds environments through gym.make; every enabled env must survive it."""
        import gymnasium as gym
        for _, spec in train.enabled(self.main):
            for mode in ("do", "ctf"):
                name = f"Task6Test-{spec['id']}-{mode}-v0"
                gym.register(name, entry_point=lambda s=spec, m=mode: train.make_env(s, m),
                             disable_env_checker=True, order_enforce=False)
                env = gym.wrappers.RecordEpisodeStatistics(gym.make(name))
                self.assertIsNotNone(env.spec)
                env.reset(seed=1)
                env.close()

    def test_cartpole_calm_intention_is_state_rule(self):
        _, spec = train.spec_of(self.main, "cartpole_calm")
        env = train.make_env(spec, "ctf")
        _, info = env.reset(seed=3)
        w = spec["behavior"]["weights"]
        for _ in range(50):
            s = train.native_scm(env).observation()
            self.assertEqual(info["next_natural_action"], int(sum(a * b for a, b in zip(w, s)) >= 0))
            _, _, term, _, info = env.step(int(info["next_natural_action"]))
            if term:
                break
        env.close()

    def test_preflight(self):
        report = train.preflight(self.main, episodes=2)
        self.assertEqual(len(report), 7)

    def test_cleanrl_arguments(self):
        _, spec = train.spec_of(self.main, "drone_r6_h10")
        job = {"algorithm": "c51", "seed": 1000, "steps": 1048576}
        argv = train.cleanrl_argv(self.main, spec, job, 5)
        self.assertIn("--n-atoms", argv)
        self.assertEqual(argv[argv.index("--v-min") + 1], "-0.1")
        ppo = train.cleanrl_argv(self.main, spec, {**job, "algorithm": "ppo"}, 5)
        self.assertIn("--no-anneal-lr", ppo)
        self.assertEqual(ppo[ppo.index("--ent-coef") + 1], "0.02")
        self.assertEqual(argv[argv.index("--seed") + 1], ppo[ppo.index("--seed") + 1])

    def test_xor_bandit(self):
        cfg = copy.deepcopy(self.smoke)
        with tempfile.TemporaryDirectory() as d:
            for algo in train.BANDITS:
                job = {"algorithm": algo, "seed": 1000, "steps": cfg["xor"]["steps"]}
                res = train.run_bandit(cfg, job, Path(d))
                self.assertEqual(set(res["checkpoints"]), {"100", "500"})
                if algo == "ucb":
                    self.assertEqual(res["own_information_regret"], 0.0)

    def test_checkpoint_hook_end_to_end(self):
        """Tiny DQN/PPO runs: every checkpoint saved at the right step and training ran."""
        import torch
        cfg = self.smoke
        index, spec = train.spec_of(cfg, "robotwalk")
        cwd = os.getcwd()
        try:
            for algo in ("dqn", "ppo"):
                with tempfile.TemporaryDirectory() as d:
                    folder = Path(d)
                    job = {"algorithm": algo, "mode": "ctf", "seed": 1000, "steps": 4096}
                    model, cps = train.train_neural(cfg, spec, index, job, folder)
                    files = sorted(p.name for p in (folder / "checkpoints").iterdir())
                    self.assertEqual(files, [f"step_{c:08d}.pt" for c in cps])
                    first = torch.load(folder / "checkpoints" / files[0])
                    last = torch.load(folder / "checkpoints" / files[-1])
                    self.assertTrue(any(not torch.equal(first[k], last[k]) for k in first))
                    self.assertTrue((folder / "scalars.csv.gz").exists())
                    self.assertFalse((folder / "runs").exists())
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
