"""Task 7 launch checks on the maze environments (python -m unittest experiments/task7/test_task7_env.py).

Covers the opt-in Causal-Gymnasium flags used by the Task 7 runner:
  seed_base_reset    - a reset seed fixes the whole start state;
  success_after_step - `terminated` uses the post-step position, like the reward;
  playback=False     - no OGBench dataset is loaded; see() needs an explicit policy.
"""
import unittest

import numpy as np
from causal_gym import AntMazePCH, HumanoidMazePCH

ENVS = {
    "ant": (AntMazePCH, "antmaze-medium-navigate-singletask-task1-v0", 1000),
    "hum": (HumanoidMazePCH, "humanoidmaze-medium-navigate-singletask-task1-v0", 2000),
}


def make(name, **kw):
    cls, env_id, H = ENVS[name]
    return cls(env_id=env_id, num_steps=H, expert_mode=False, seed=0, playback=False, **kw)


def actions(env, n, seed):
    rng = np.random.default_rng(seed)
    space = env.env.action_space
    return [rng.uniform(space.low, space.high).astype(np.float32) for _ in range(n)]


def state(obs):
    """Latest public state, excluding the action history X (empty before the first step)."""
    return np.concatenate([np.ravel(v[-1]) for k, v in sorted(obs.items()) if k != "X"])


def rollout(env, seed, acts):
    obs, info = env.reset(seed=seed)
    traj = [state(obs)]
    out = []
    for a in acts:
        o, r, term, trunc, _ = env.do(lambda _o, a=a: a)
        traj.append(state(o))
        out.append((r, term, trunc))
        if term or trunc:
            break
    return np.stack(traj), out, info


class SeedBaseReset(unittest.TestCase):
    def test_same_seed_same_trajectory(self):
        for name in ENVS:
            with self.subTest(env=name):
                e1, e2 = make(name, seed_base_reset=True), make(name, seed_base_reset=True)
                acts = actions(e1, 40, 1)
                # Burn a reset on e2 so both env objects are at different RNG positions.
                e2.reset(seed=999)
                t1, o1, _ = rollout(e1, 123, acts)
                t2, o2, _ = rollout(e2, 123, acts)
                np.testing.assert_array_equal(t1, t2)
                self.assertEqual(o1, o2)
                t3, _, _ = rollout(e1, 124, acts)
                self.assertFalse(np.array_equal(t1[0], t3[0]), "different seeds should differ")

    def test_default_keeps_unseeded_base(self):
        # Without the flag the start state depends on the env's reset history (original behavior).
        e1, e2 = make("ant"), make("ant")
        e2.reset(seed=999)
        t1, _, _ = rollout(e1, 123, [])
        t2, _, _ = rollout(e2, 123, [])
        self.assertFalse(np.array_equal(t1[0], t2[0]))

    def test_goal_constant_across_resets(self):
        for name in ENVS:
            with self.subTest(env=name):
                env = make(name, seed_base_reset=True)
                for s in range(20):
                    _, info = env.reset(seed=s)
                    np.testing.assert_allclose(info["env_info"]["goal"][:2], env.env._goal_xy)


class SuccessAfterStep(unittest.TestCase):
    def check(self, name, fixed):
        env = make(name, seed_base_reset=True, success_after_step=fixed)
        acts = actions(env, 30, 2)
        rollout(env, 7, acts)                      # learn where the agent is after step k
        k = 12
        target = np.asarray(env.env.P[k + 1][:2])  # P[0] is the reset state, P[k+1] after action k
        env.env._goal_xy = target
        env.env.success_radius = 1e-9
        _, out, _ = rollout(env, 7, acts)
        first = next(i for i, (_, term, _) in enumerate(out) if term) if any(o[1] for o in out) else None
        rewards = [r for r, _, _ in out]
        return first, rewards

    def test_terminated_on_entering_goal(self):
        for name in ENVS:
            with self.subTest(env=name):
                first, rewards = self.check(name, fixed=True)
                self.assertEqual(first, 12)
                self.assertGreater(rewards[12], rewards[11] + 0.5)  # reward credits the same step

    def test_default_lags_one_step(self):
        first, _ = self.check("ant", fixed=False)
        self.assertEqual(first, 13)


class Playback(unittest.TestCase):
    def test_no_dataset_and_explicit_policy(self):
        for name in ENVS:
            with self.subTest(env=name):
                env = make(name)
                self.assertIsNone(env.expert)
                env.reset(seed=1)
                with self.assertRaises(ValueError):
                    env.see()


if __name__ == "__main__":
    unittest.main()
