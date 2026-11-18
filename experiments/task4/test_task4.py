"""Task 4 protocol, original-function integration and evidence-integrity checks."""
import copy
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as train

train.source_paths()


class Task4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = train.configuration(HERE/"config.json")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.template = Path(cls.temporary.name)/"template"
        cls.template.mkdir()
        cls.cfg = copy.deepcopy(train.configuration(HERE/"config.json", "smoke"))
        cls.cfg["environments"] = [cls.cfg["environments"][2]]
        train.freeze(cls.template, cls.cfg, "test", train.source_paths())
        train.prepare_datasets(cls.template)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def bundle(self, folder):
        shutil.copytree(self.template, folder)
        return folder

    def records(self, spec=None, count=128):
        spec = spec or self.main["environments"][2]
        env = train.make_environment(spec)
        actions, rewards = [], []
        try:
            for seed in train.episode_seeds(2, 42, 0, count):
                env.reset(seed=int(seed))
                _, reward, _, _, info = env.see()
                actions.append(info["natural_action"])
                rewards.append(reward)
        finally:
            env.close()
        a, r = np.array(actions), np.array(rewards)
        return a, r, train.bound_statistics(a, r, .05)

    def test_approved_protocol_and_unique_cells(self):
        c = self.main
        self.assertEqual(len(c["environments"])*len(c["methods"])*len(c["seeds"]), 420)
        self.assertEqual(420*c["online_steps"], 4200000)
        self.assertEqual(c["offline_samples"], 5000)
        self.assertEqual(sum(e["kind"] == "mabcool" for e in c["environments"]), 3)
        self.assertEqual(sum(e["kind"] == "mab" for e in c["environments"]), 4)
        self.assertEqual(c["seeds"], list(range(2000, 2020)))

    def test_native_factory_and_bound_separation(self):
        from causal_gym.envs.mab_cool import MABCOOLPCH
        from causal_gym.envs.mab import MABPCH
        for spec in self.main["environments"]:
            env = train.make_environment(spec)
            self.assertIsInstance(env, MABCOOLPCH if spec["kind"] == "mabcool" else MABPCH)
            self.assertGreater(spec["rho"], 0)
            d = train.theoretical_diagnostics(spec, self.main)
            self.assertGreater(d["robust_separation_margin"], 0)
            if spec["kind"] == "mabcool":
                self.assertEqual(env.env.delta, spec["delta"])
                self.assertEqual(env.env.action(.79), 1)
                self.assertEqual(env.env.action(.81), 0)
            else:
                for u in (.0, spec["q"]-.001, spec["q"]+.001, .999):
                    expected = int(u >= spec["q"]) if spec["mirror"] else int(u < spec["q"])
                    self.assertEqual(env.env._policy(u), expected)
            env.close()

    def test_mirror_swaps_both_mechanisms(self):
        forward = self.main["environments"][3]
        mirror = self.main["environments"][4]
        a, b = train.make_environment(forward), train.make_environment(mirror)
        try:
            for seed in range(100):
                a.reset(seed=seed)
                b.reset(seed=seed)
                xa, ya = a.see()[4]["natural_action"], a.env.y
                xb, yb = b.see()[4]["natural_action"], b.env.y
                self.assertEqual(xa, 1-xb)
                self.assertEqual(ya, yb)
                for action in (0, 1):
                    a.reset(seed=seed)
                    b.reset(seed=seed)
                    self.assertEqual(a.do(lambda: action)[1], b.do(lambda: 1-action)[1])
        finally:
            a.close()
            b.close()

    def test_joint_bounds_and_initialization_match_formulas(self):
        actions = np.array([0, 0, 1, 1])
        rewards = np.array([0, 0, 0, 1])
        b = train.bound_statistics(actions, rewards, .05)
        np.testing.assert_array_equal(b["joint"], [[.5, 0], [.25, .25]])
        eps = np.sqrt(np.log(160)/8)
        np.testing.assert_allclose(b["lower"], np.maximum(0, np.array([0, .25])-eps))
        np.testing.assert_allclose(b["upper"], np.minimum(1, np.array([.5, .75])+eps))
        np.testing.assert_allclose(b["direct_initial_indices"], np.array([0, 1])/3+np.sqrt(np.log(1e4)/3))

    def test_actual_repository_functions_are_called(self):
        import causal_rl.algo.cool.cool as local
        spec = self.main["environments"][2]
        a, r, bounds = self.records(spec)
        seeds = train.episode_seeds(2, 42, 1, 64)
        with mock.patch.object(local, "cool_mab_ucb", wraps=local.cool_mab_ucb) as cool, \
             mock.patch.object(local, "mab_ucb_direct_transfer", wraps=local.mab_ucb_direct_transfer) as direct:
            traces = {m: train.execute_local(spec, m, a, r, seeds, bounds) for m in train.METHODS}
            self.assertEqual(cool.call_count, 2)
            self.assertIsNone(cool.call_args_list[0].kwargs["lower"])
            np.testing.assert_array_equal(cool.call_args_list[1].kwargs["upper"], bounds["upper"])
            self.assertEqual(direct.call_count, 1)
            self.assertEqual(direct.call_args.kwargs["see_timesteps"], len(a))
            self.assertEqual(traces["direct"].offline_index, len(a))

    def test_determinism_all_seven_configurations(self):
        for index, spec in enumerate(self.main["environments"]):
            a, r, bounds = self.records(spec)
            seeds = train.episode_seeds(index, 20, 1, 32)
            for method in train.METHODS:
                x = train.execute_local(spec, method, a, r, seeds, bounds)
                y = train.execute_local(spec, method, a, r, seeds, bounds)
                self.assertEqual(x.actions, y.actions)
                self.assertEqual(x.rewards, y.rewards)

    def test_recovery_is_identical_without_replaying_environment_interactions(self):
        spec = self.main["environments"][2]
        a, r, bounds = self.records(spec)
        seeds = train.episode_seeds(2, 42, 1, 64)
        for method in train.METHODS:
            fresh = train.execute_local(spec, method, a, r, seeds, bounds)
            env = train.make_environment(spec)
            with mock.patch.object(env, "do", wraps=env.do) as calls, \
                 mock.patch.object(train, "make_environment", return_value=env):
                resumed = train.execute_local(spec, method, a, r, seeds, bounds,
                    prefix=(fresh.actions[:37], fresh.rewards[:37]))
                self.assertEqual(calls.call_count, 64-37)
            self.assertEqual(fresh.actions, resumed.actions)
            self.assertEqual(fresh.rewards, resumed.rewards)

    def test_saved_checkpoint_pause_resume_and_shared_data(self):
        with tempfile.TemporaryDirectory() as temp:
            fresh = self.bundle(Path(temp)/"fresh")
            resumed = self.bundle(Path(temp)/"resumed")
            env_id = self.cfg["environments"][0]["id"]
            for method in train.METHODS:
                train.run_job(str(fresh), env_id, method, 42)
                result = train.run_job(str(resumed), env_id, method, 42, stop_after=37)
                self.assertEqual(result["status"], "paused")
                self.assertEqual(result["steps"], 37)
                train.run_job(str(resumed), env_id, method, 42)
                x, y = (train.job_path(root, env_id, method, 42) for root in (fresh, resumed))
                self.assertEqual(train.read_json(x/"summary.json"), train.read_json(y/"summary.json"))
                with np.load(x/"trace.npz") as u, np.load(y/"trace.npz") as v:
                    for field in u.files:
                        np.testing.assert_array_equal(u[field], v[field])
            summaries = [train.read_json(train.job_path(fresh, env_id, m, 42)/"summary.json")
                         for m in train.METHODS]
            self.assertEqual(len({s["offline_sha256"] for s in summaries}), 1)

    def test_checkpoint_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.bundle(Path(temp)/"run")
            env_id = self.cfg["environments"][0]["id"]
            train.run_job(str(root), env_id, "direct", 42, stop_after=37)
            path = train.job_path(root, env_id, "direct", 42)/"trace.npz"
            with np.load(path) as original:
                data = {k: original[k].copy() for k in original.files}
            data["rewards"][0] = 1-data["rewards"][0]
            train.atomic_write(path, lambda f: np.savez_compressed(f, **data))
            with self.assertRaisesRegex(ValueError, "checksum"):
                train.run_job(str(root), env_id, "direct", 42)

    def test_no_checkpoint_does_not_silently_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.bundle(Path(temp)/"run")
            env_id = self.cfg["environments"][0]["id"]
            train.run_job(str(root), env_id, "ucb", 42, stop_after=16)
            (train.job_path(root, env_id, "ucb", 42)/"trace.npz").unlink()
            with self.assertRaisesRegex(ValueError, "no silent restart"):
                train.run_job(str(root), env_id, "ucb", 42)


    def test_snapshot_tampering_and_task5_manifests_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.bundle(Path(temp)/"run")
            source = root/"snapshot/causal_rl/algo/cool/cool.py"
            source.write_text(source.read_text()+"\n# changed\n")
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                train.verify_snapshot(root)
            meta = train.read_json(root/"manifest.json")
            meta["task"] = 5
            train.write_json(root/"manifest.json", meta)
            with self.assertRaisesRegex(ValueError, "Task 5"):
                train.verify_snapshot(root)

    def test_task5_output_rejected_before_any_file_is_written(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"task5"
            root.mkdir()
            train.write_json(root/"manifest.json", {"identity": "existing-task5", "config": {}})
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with self.assertRaisesRegex(ValueError, "Task 5"):
                train.main(["prepare", "--output", str(root)])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_oracle_metrics_support_both_optimal_arm_directions(self):
        for spec in (self.main["environments"][1], self.main["environments"][4]):
            best = int(np.argmax(spec["means"]))
            actions = [best]*64
            cfg = {**self.cfg, "checkpoints": [64]}
            bounds = train.bound_statistics(np.array([0, 1]), np.array([0, 1]), .05)
            metrics = train.metrics_for_trace(spec, "ucb", actions, [1]*64, bounds, cfg)[0]
            self.assertEqual(metrics["cumulative_pseudo_regret"], 0)
            self.assertEqual(metrics["simple_regret"], 0)
            self.assertEqual(metrics["normalized_score"], 1)
            self.assertEqual(metrics["optimal_arm_fraction"], 1)

    def test_seed_streams_are_separate_and_reproducible(self):
        samples = []
        for env in range(7):
            for seed in range(2000, 2020):
                for stream in (0, 1):
                    x = train.episode_seeds(env, seed, stream, 10)
                    np.testing.assert_array_equal(x, train.episode_seeds(env, seed, stream, 10))
                    samples.extend(map(int, x))
        self.assertEqual(len(samples), len(set(samples)))




if __name__ == "__main__":
    unittest.main()
