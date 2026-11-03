"""Native causal coupling, Gym lifecycle, and actual DeepRL integration."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ctf-wrapper-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ctf-wrapper-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import unittest
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatten
import numpy as np

from causal_rl.wrappers import CTFWrapper
from causal_gym.envs.mdp import MDPPCH
from causal_gym.envs.robowalk import RobotWalkPCH
from causal_gym.envs.rescue_drone import RescueDronePCH
from causal_gym.envs.cartpole_wind import CartPoleWindPCH


class CallbackEnv(gym.Env):
    """A latent is generated only inside ctf_do, never at reset or in step."""
    observation_space = spaces.Discrete(2)
    action_space = spaces.Discrete(2)

    def __init__(self, truncate_at=None, fail=False):
        self.truncate_at = truncate_at
        self.fail = fail
        self.transitions = 0
        self.draws = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.transitions = 0
        self.draws = []
        return self.state, {"reset_option": options}

    def ctf_do(self, policy):
        u = int(self.np_random.integers(2))
        self.draws.append(u)
        intention = self.state ^ u
        action = policy(self.state, intention)
        if self.fail:
            raise ValueError("native failure")
        self.state = u
        self.transitions += 1
        return self.state, float(action == u), False, self.transitions == self.truncate_at, {
            "natural_action": intention, "action": action,
        }

    def do(self, policy):
        return self.ctf_do(lambda state, intention: policy(state))

    def step(self, action):
        raise AssertionError("The adapter must never bypass native do/ctf_do")


def make_vector_env():
    return CTFWrapper(MDPPCH(max_step=4), horizon=4)


class CTFWrapperTests(unittest.TestCase):
    def test_pending_noise_and_next_intention_are_the_executed_ones(self):
        native = CallbackEnv()
        env = CTFWrapper(native, horizon=4)
        obs, info = env.reset(seed=30, options={"example": True})
        self.assertEqual(native.transitions, 0)
        self.assertEqual(len(native.draws), 1)
        self.assertEqual(info["reset_option"], {"example": True})
        for t in range(4):
            self.assertTrue(env.observation_space.contains(obs))
            state = int(obs[:2].argmax())
            intention = int(obs[2:4].argmax())
            current_u = native.draws[-1]
            self.assertEqual(intention, state ^ current_u)
            previous_obs, snapshot = obs, obs.copy()
            obs, reward, terminated, truncated, info = env.step(state ^ intention)
            np.testing.assert_array_equal(previous_obs, snapshot)  # No future state/intention aliasing.
            self.assertEqual(reward, 1.)  # A acts on the same U that generated I.
            self.assertEqual(info["natural_action"], intention)  # I_t, not I_{t+1}.
            self.assertEqual(native.transitions, t + 1)
            self.assertFalse(truncated)
            self.assertEqual(terminated, t == 3)
        np.testing.assert_array_equal(obs[2:4], [0, 0])
        self.assertIsNone(info["next_natural_action"])
        self.assertEqual(len(native.draws), 4)  # No fictitious post-terminal intention.
        env.close()

    def test_native_pch_trajectory_equivalence(self):
        factories = [
            (lambda: MDPPCH(max_step=6), 6),
            (lambda: RobotWalkPCH(length=3, timelimit=5), 6),
            (lambda: RescueDronePCH(grid_size=5, max_steps=4), 4),
            (lambda: RescueDronePCH(grid_size=5, random_start=True, max_steps=4), 4),
            (lambda: CartPoleWindPCH(max_episode_steps=6), 6),
        ]
        for factory, horizon in factories:
            for mode in ("ctf", "do"):
                for seed in range(12):
                    native = factory(); env = CTFWrapper(factory(), mode=mode, horizon=horizon)
                    native.reset(seed=seed); obs, _ = env.reset(seed=seed)
                    n = int(env.action_space.n)
                    for t in range(horizon):
                        action = int(obs[-n-1:-1].argmax()) if mode == "ctf" else t % n
                        def policy(state, intention=None):
                            expected = flatten(native.observation_space, state).astype(np.float32)
                            np.testing.assert_array_equal(obs[:len(expected)], expected)
                            if mode == "ctf":
                                self.assertEqual(action, int(intention))
                            return action
                        direct = native.ctf_do(policy) if mode == "ctf" else native.do(policy)
                        nxt, reward, term, trunc, info = env.step(action)
                        self.assertEqual(float(direct[1]), reward)
                        expected = flatten(native.observation_space, direct[0]).astype(np.float32)
                        np.testing.assert_array_equal(nxt[:len(expected)], expected)
                        self.assertEqual(term, bool(direct[2] or t + 1 == horizon))
                        self.assertFalse(trunc)
                        if mode == "ctf": self.assertEqual(info['natural_action'], direct[4]['natural_action'])
                        obs = nxt
                        if term: break
                    native.close(); env.close()

    def test_do_control_has_no_intention_and_same_fixed_action_outcomes(self):
        do = CTFWrapper(MDPPCH(max_step=5), mode="do", horizon=5)
        ctf = CTFWrapper(MDPPCH(max_step=5), horizon=5)
        a, _ = do.reset(seed=25); b, _ = ctf.reset(seed=25)
        self.assertEqual(a.shape, (3,)); self.assertEqual(b.shape, (5,))
        for action in (0, 1, 1, 0, 1):
            a, ra, ta, tra, _ = do.step(action)
            b, rb, tb, trb, _ = ctf.step(action)
            np.testing.assert_array_equal(a[:2], b[:2])
            self.assertEqual(a[-1], b[-1]); self.assertEqual((ra,ta,tra), (rb,tb,trb))
        do.close(); ctf.close()

    def test_reset_and_close_cancel_without_executing_pending_actions(self):
        native = CallbackEnv(); env = CTFWrapper(native, horizon=3)
        first, _ = env.reset(seed=17)
        again, _ = env.reset(seed=17)
        np.testing.assert_array_equal(first, again)
        self.assertEqual(native.transitions, 0)
        env.close()
        self.assertEqual(native.transitions, 0)
        with self.assertRaises(gym.error.ResetNeeded): env.step(0)

    def test_invalid_actions_do_not_consume_a_transition(self):
        native = CallbackEnv(); env = CTFWrapper(native, horizon=1)
        with self.assertRaises(gym.error.ResetNeeded): env.step(0)
        env.reset(seed=2)
        prepared_draws = native.draws.copy()
        for invalid in (7, -1, 9):
            with self.assertRaises(ValueError): env.step(invalid)
            self.assertEqual(native.draws, prepared_draws)  # Retry retains the prepared U.
        self.assertEqual(native.transitions, 0)
        self.assertTrue(env.step(0)[2])
        with self.assertRaises(gym.error.ResetNeeded): env.step(0)
        env.close()

    def test_finite_horizon_masks_terminal_padding(self):
        env = CTFWrapper(CallbackEnv(truncate_at=2), horizon=2)
        env.reset(seed=2); env.step(0)
        obs, _, term, trunc, info = env.step(0)
        self.assertTrue(term); self.assertFalse(trunc)
        self.assertFalse(info['native_terminated']); self.assertTrue(info['native_truncated'])
        self.assertTrue(info['task_horizon_reached'])
        np.testing.assert_array_equal(obs[2:], [0.,0.,0.])
        env.close()

    def test_unknown_ctf_truncation_never_silently_bootstraps_zero_intention(self):
        env = CTFWrapper(CallbackEnv(truncate_at=1))
        env.reset(seed=2)
        with self.assertRaisesRegex(RuntimeError, 'no next intention'): env.step(0)
        env.close()
        do = CTFWrapper(CallbackEnv(truncate_at=1), mode="do")
        do.reset(seed=2)
        obs, _, term, trunc, _ = do.step(0)
        self.assertFalse(term); self.assertTrue(trunc); self.assertTrue(do.observation_space.contains(obs))
        do.close()

    def test_native_errors_propagate_and_allow_reset(self):
        native = CallbackEnv(fail=True); env = CTFWrapper(native, horizon=2)
        env.reset(seed=2)
        with self.assertRaisesRegex(ValueError, 'native failure'): env.step(0)
        native.fail = False
        env.reset(seed=2); env.step(0); env.close()

    def test_sync_and_process_vector_envs(self):
        for kind in ('sync','spawn'):
            vector = (gym.vector.SyncVectorEnv([make_vector_env] * 2) if kind == 'sync'
                      else gym.vector.AsyncVectorEnv([make_vector_env] * 2, context='spawn'))
            try:
                obs, _ = vector.reset(seed=31)
                self.assertEqual(obs.shape, (2, 5))
                for _ in range(9):
                    obs, rewards, terminated, truncated, info = vector.step(np.array([0,1]))
                    self.assertTrue(np.isfinite(obs).all()); self.assertFalse(truncated.any())
            finally:
                vector.close()

    def test_dqn_and_ppo_can_train_without_experiment_adapters(self):
        import torch
        from stable_baselines3 import DQN, PPO
        torch.set_num_threads(1)
        for learner in (DQN, PPO):
            env = CTFWrapper(MDPPCH(max_step=4), horizon=4)
            kwargs = dict(policy_kwargs={"net_arch": [16]}, seed=3, device="cpu", verbose=0)
            if learner is DQN:
                kwargs.update(learning_starts=4, buffer_size=64, batch_size=4, train_freq=1, gradient_steps=1)
            else:
                kwargs.update(n_steps=8, batch_size=8, n_epochs=1)
            model = learner("MlpPolicy", env, **kwargs)
            before = [p.detach().clone() for p in model.policy.parameters()]
            model.learn(total_timesteps=32)
            self.assertEqual(model.num_timesteps, 32)
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.policy.parameters())))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
            env.close()


if __name__ == "__main__":
    unittest.main()
