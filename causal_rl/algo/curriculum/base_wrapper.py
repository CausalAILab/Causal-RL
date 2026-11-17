# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/environments/base_wrapper.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
import numpy as np
import torch
import gymnasium as gym

from causal_rl.algo.curriculum.teachers.dummy_teachers import DistributionSampler, UniformSampler


class BaseWrapper(gym.Env):
    """
    Contextual environment driven by a curriculum teacher.

    On every reset a context is drawn from `teacher.sample()` and written into the raw
    environment via `env.unwrapped.set_context`. If target bounds are given, curriculum teachers
    may only propose source (non-target) tasks: target contexts are rejected and, if the teacher
    keeps proposing them, `non_target_sampler` is used instead. Without target bounds
    (`--allow-target-task`) every teacher sample is used. `DistributionSampler` /
    `UniformSampler` (the default curriculum and evaluation) are used as-is. When an episode
    ends, `done_callback` (implemented by the teacher-specific subclasses) reports the episode
    context and its outcome to the teacher.
    """

    def __init__(self, env, teacher, discount_factor, target_ctx_upper=None, target_ctx_lower=None,
                 non_target_sampler=None):
        gym.Env.__init__(self)
        self.env = env
        self.teacher = teacher
        self.discount_factor = discount_factor
        self.target_ctx_upper = target_ctx_upper
        self.target_ctx_lower = target_ctx_lower
        self.non_target_sampler = non_target_sampler

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        if hasattr(self.env, "reward_range"):
            self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

        self.discounted_reward = 0.
        self.cur_disc = 1.
        self.cur_context = None

        # all contexts used for training (saved to clist-<env>.json after training)
        self.train_contexts = []

        # rendered frames of the current episode and summary of the last finished one
        self.img_states = []
        self.last_episode_vis = None

    def seed(self, seed):
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        self.env.unwrapped.np_random = np.random.default_rng(seed)

    def done_callback(self, step, context, discounted_reward):
        pass

    def step(self, action):
        step = self.env.step(action)
        self.img_states.append(self.render())
        self.update(step)
        return step

    def sample_non_target_ctx(self, patience=10):
        """
        Keep sampling from the teacher until the context is not in the target range.
        If the teacher keeps sampling target contexts, fall back to `non_target_sampler`.
        Without target bounds (`--allow-target-task`) the first teacher sample is returned as-is.
        """
        if self.target_ctx_upper is None or isinstance(self.teacher, (DistributionSampler, UniformSampler)):
            return self.teacher.sample()
        for _ in range(patience):
            context = self.teacher.sample()
            if np.all(context < self.target_ctx_upper) and \
                    np.all(context >= self.target_ctx_lower):
                continue
            return context
        return self.non_target_sampler()

    def reset(self, seed=None, options={}):
        self.cur_context = self.sample_non_target_ctx()
        self.env.unwrapped.set_context(self.cur_context.copy())
        obs, info = self.env.reset(seed=seed, options=options)

        self.train_contexts.append(self.cur_context.tolist())
        self.img_states = [self.render()]

        return obs, info

    def render(self):
        return self.env.render()

    def update(self, step):
        # step = obs, reward, term, trunc, info
        reward = step[1]
        self.discounted_reward += self.cur_disc * reward
        self.cur_disc *= self.discount_factor

        assert isinstance(step[2], bool) and isinstance(step[3], bool), \
            "step[2] and step[3] should be the boolean 'terminated' and 'truncated' signals."
        if step[2] or step[3]:
            self.done_callback(step, self.cur_context, self.discounted_reward)

            # Episode video in N (=1), T, C, H, W format
            imgs = torch.permute(torch.tensor(np.array(self.img_states)), (0, 3, 1, 2)).unsqueeze(0)
            # Last three entries of the (env-processed) context:
            # Colored Sokoban - box position, agent position, intervention
            # Button Maze     - fix button, fix light, agent position
            context = self.env.unwrapped.context
            self.last_episode_vis = {
                'context_0': context[-3],
                'context_1': context[-2],
                'context_2': context[-1],
                'imgs': imgs,
            }

            self.img_states = []
            self.discounted_reward = 0.
            self.cur_disc = 1.
            self.cur_context = None

    def get_vis(self):
        if self.last_episode_vis is not None:
            return self.last_episode_vis
        else:
            return {}
