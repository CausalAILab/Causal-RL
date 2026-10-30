"""Protocol, oracle separation, recovery, and statistical-estimator regressions."""
import copy
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import task5 as train
import task5_report as report

train.source_paths()


class Task5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = train.configuration(ROOT / "configs/task5.json")
        cls.plans = {e["id"]: train.prepare_environment(e) for e in cls.cfg["environments"]}

    def bundle(self, root, env_id="example_9_5", seeds=(20, 21), budget=96):
        plan = copy.deepcopy(self.plans[env_id])
        plan["spec"]["budget"] = budget
        plan["spec"]["checkpoints"] = [16, 32, budget]
        cfg = copy.deepcopy(self.cfg)
        cfg.update(environments=[plan["spec"]], seeds=list(seeds), common_steps=32,
                   checkpoint_every=16, optimal_arm_window=32)
        identity = {"config": cfg, "mode": "test", "sources": {}, "runtime": train.runtime_versions()}
        saved = {**identity, "identity": train.digest(identity)}
        train.write_json(root / "manifest.json", saved)
        train.write_json(root / "snapshot/configs/task5.json", cfg)
        train.write_json(root / "environments" / (env_id + ".json"), plan)
        train.write_json(root / "preflight.json", {"identity": saved["identity"],
                         "environment_hashes": {env_id: train.digest(plan)}})
        return plan

    def test_main_protocol_size_budgets_and_arm_counts(self):
        self.assertEqual(self.cfg["seeds"], list(range(1000, 1020)))
        self.assertEqual(len(self.plans)*4*len(self.cfg["seeds"]), 880)
        self.assertEqual(sum(e["budget"] for e in self.cfg["environments"])*80, 12400000)
        for e in self.cfg["environments"]:
            self.assertEqual(e["budget"], 25000 if e["d"] == 6 else 10000)
            self.assertEqual([len(self.plans[e["id"]]["candidates"][m]) for m in train.METHODS],
                             e["expected_counts"])

    def test_oracles_and_legality(self):
        example = self.plans["example_9_5"]
        self.assertAlmostEqual(example["optimum"], 1)
        self.assertAlmostEqual(example["set_optima"]["all_at_once"], .5)
        six = self.plans["six_variable"]
        self.assertAlmostEqual(six["optimum"], .6332843008)
        self.assertAlmostEqual(six["set_optima"]["all_at_once"], .53404)
        for method, ids in six["candidates"].items():
            for i in ids:
                self.assertLessEqual(set(six["arms"][i]), {"B", "D", "E"})
        for plan in self.plans.values():
            for m in ("mis", "pomis"):
                self.assertAlmostEqual(plan["set_optima"][m], plan["optimum"])
        for d in (2, 4, 6):
            p = self.plans[f"collider_d{d}"]
            self.assertEqual(p["candidates"]["full"], p["candidates"]["mis"])

    def test_reward_sampling_agrees_with_six_variable_oracle(self):
        plan = self.plans["six_variable"]
        arm = int(np.argmax(plan["means"]))
        env = train.make_environment(plan["spec"])
        rng = np.random.default_rng(29)
        rewards = [train.execute_intervention(env, plan["spec"], plan["arms"][arm],
                   int(rng.integers(0, 2**64, dtype=np.uint64))) for _ in range(5000)]
        self.assertLess(abs(np.mean(rewards)-plan["means"][arm]), .035)

    def test_resume_is_identical_to_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            fresh, resumed = Path(temporary)/"fresh", Path(temporary)/"resumed"
            self.bundle(fresh)
            self.bundle(resumed)
            train.run_job(fresh, "example_9_5", "full", 20)
            state = train.run_job(resumed, "example_9_5", "full", 20, stop_after=37)
            self.assertEqual(state["status"], "paused")
            self.assertEqual(state["steps"], 37)
            train.run_job(resumed, "example_9_5", "full", 20)
            a = train.job_path(fresh, "example_9_5", "full", 20)
            b = train.job_path(resumed, "example_9_5", "full", 20)
            with np.load(a/"checkpoint.npz", allow_pickle=False) as x, np.load(b/"checkpoint.npz", allow_pickle=False) as y:
                for field in x.files:
                    np.testing.assert_array_equal(x[field], y[field], err_msg=field)
            self.assertEqual(train.read_json(a/"summary.json"), train.read_json(b/"summary.json"))

    def test_shared_noise_pairing_and_regret_decomposition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = self.bundle(root)
            for method in ("full", "all_at_once"):
                train.run_job(root, "example_9_5", method, 20)
            paths = [train.job_path(root, "example_9_5", m, 20) for m in ("full", "all_at_once")]
            with np.load(paths[0]/"checkpoint.npz") as a, np.load(paths[1]/"checkpoint.npz") as b:
                np.testing.assert_array_equal(a["episode_seeds"], b["episode_seeds"])
                self.assertEqual(b["episode_seeds"].dtype, np.dtype("uint64"))
            for p in paths:
                for row in train.read_json(p/"summary.json")["metrics"]:
                    self.assertAlmostEqual(row["cumulative_pseudo_regret"], row["own_set_regret"]+
                                           row["steps"]*row["candidate_set_value_gap"])

    def test_scores_cannot_change_learner_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            real, changed = Path(temporary)/"real", Path(temporary)/"changed"
            self.bundle(real)
            plan = self.bundle(changed)
            plan["means"] = (.1+.8*(1-np.asarray(plan["means"]))).tolist()
            plan["optimum"], plan["minimum"] = max(plan["means"]), min(plan["means"])
            plan["set_optima"] = {m: max(plan["means"][a] for a in ids) for m, ids in plan["candidates"].items()}
            train.write_json(changed/"environments/example_9_5.json", plan)
            preflight = train.read_json(changed/"preflight.json")
            preflight["environment_hashes"]["example_9_5"] = train.digest(plan)
            train.write_json(changed/"preflight.json", preflight)
            for root in (real, changed):
                train.run_job(root, "example_9_5", "full", 20)
            pa, pb = (train.job_path(r, "example_9_5", "full", 20)/"checkpoint.npz" for r in (real, changed))
            with np.load(pa) as a, np.load(pb) as b:
                for field in ("actions", "rewards", "counts", "values", "episode_seeds"):
                    np.testing.assert_array_equal(a[field], b[field])

    def test_corrupt_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.bundle(root)
            train.run_job(root, "example_9_5", "full", 20, stop_after=37)
            checkpoint = train.job_path(root, "example_9_5", "full", 20)/"checkpoint.npz"
            with np.load(checkpoint) as original:
                payload = {k: original[k].copy() for k in original.files}
            payload["values"][:] = .123
            train.atomic_write(checkpoint, lambda f: np.savez_compressed(f, **payload))
            with self.assertRaisesRegex(ValueError, "counts/values"):
                train.run_job(root, "example_9_5", "full", 20)

    def test_missing_runs_are_not_zero_imputed_or_ranked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.bundle(root)
            train.run_job(root, "example_9_5", "full", 20)
            self.assertEqual(report.generate(root, quick=True), 2)
            path = root/"reports/diagnostic"
            manifest = train.read_json(path/"report_manifest.json")
            self.assertEqual(manifest["completed_runs"], 1)
            self.assertEqual(manifest["planned_runs"], 8)
            self.assertEqual(len(manifest["missing_runs"]), 7)
            self.assertFalse((path/"suite.csv").exists())

    def test_independent_seed_streams(self):
        samples = []
        for env_index, seed in itertools.product(range(11), range(1000, 1020)):
            rng = np.random.default_rng(np.random.SeedSequence([0xCA05A1, env_index, seed]))
            samples.extend(map(int, rng.integers(0, 2**64, size=20, dtype=np.uint64)))
        self.assertEqual(len(samples), len(set(samples)))

    def test_source_inventory_contains_import_assets(self):
        inventory = train.source_files(train.source_paths())
        self.assertIn("causal_gym/envs/assets/coin.png", inventory)
        self.assertIn("causal_rl/algo/ctf_do/ctfucb.py", inventory)


class StatisticsTests(unittest.TestCase):
    def test_iqm_trims_tails(self):
        self.assertEqual(report.iqm(np.array([0, 1, 2, 3, 4, 5, 6, 1000])), 3.5)

    def test_probability_uses_all_pairs_and_half_ties(self):
        self.assertEqual(report.probability_of_improvement(np.array([[0., 2.]]), np.array([[1., 2.]])), .375)

    def test_stratified_bootstrap_preserves_pairing_and_task_membership(self):
        base = np.array([[.1, .2, .3], [.4, .5, .6]])
        scores = np.stack((base, base+.1))
        for sample in report.stratified_samples(scores, 173, np.random.default_rng(1)):
            self.assertEqual(sample.shape[2:], (2, 3))
            np.testing.assert_allclose(sample[1]-sample[0], .1, atol=1e-15)
            self.assertTrue(np.isin(sample[0, :, 0], base[0]).all())
            self.assertTrue(np.isin(sample[0, :, 1], base[1]).all())
        point, low, high = report.suite_ci(scores, 200, np.random.default_rng(2))
        np.testing.assert_allclose(point[1]-point[0], [.1, -.1], atol=1e-15)
        np.testing.assert_allclose(low[1]-low[0], [.1, -.1], atol=1e-15)
        np.testing.assert_allclose(high[1]-high[0], [.1, -.1], atol=1e-15)

    def test_profiles_and_probability_intervals(self):
        scores = np.array([[[0., 1.], [.5, .5]], [[0., 1.], [.5, .5]]])
        thresholds = np.array([0., .5, 1.])
        point, low, high = report.profile_ci(scores, thresholds, 100, np.random.default_rng(3))
        np.testing.assert_allclose(point[0], [.75, .25, 0])
        self.assertTrue(np.all(np.diff(point, axis=-1) <= 0))
        pi, pil, pih = report.probability_ci(scores, 100, np.random.default_rng(4))
        np.testing.assert_allclose(pi, .5)
        np.testing.assert_allclose(pil, .5)
        np.testing.assert_allclose(pih, .5)

    def test_resample_entire_run_vector(self):
        scores = np.array([[[1., 3.], [2., 6.], [3., 9.]]])
        point, low, high = report.mean_ci(scores, 200, np.random.default_rng(5))
        np.testing.assert_allclose(high[:, 1], 3*high[:, 0])
        np.testing.assert_allclose(low[:, 1], 3*low[:, 0])


if __name__ == "__main__":
    unittest.main()
