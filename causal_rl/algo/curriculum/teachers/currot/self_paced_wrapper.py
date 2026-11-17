# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/teachers/spl/self_paced_wrapper.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
import numpy as np
from causal_rl.algo.curriculum.teachers.util import Buffer
from causal_rl.algo.curriculum.base_wrapper import BaseWrapper


class SelfPacedWrapper(BaseWrapper):
    """Updates the (CurrOT) teacher with the contexts and discounted returns of every
    `episodes_per_update` finished episodes."""

    def __init__(self, env, sp_teacher, discount_factor, target_ctx_upper=None, target_ctx_lower=None,
                 non_target_sampler=None, episodes_per_update=50):
        BaseWrapper.__init__(self, env, sp_teacher, discount_factor, target_ctx_upper, target_ctx_lower,
                             non_target_sampler=non_target_sampler)

        self.context_buffer = Buffer(2, episodes_per_update + 1)
        self.episodes_per_update = episodes_per_update

    def done_callback(self, step, context, discounted_reward):
        self.context_buffer.update_buffer((context, discounted_reward))

        if len(self.context_buffer) >= self.episodes_per_update:
            contexts, returns = self.context_buffer.read_buffer()
            self.teacher.update_distribution(np.array(contexts), np.array(returns))
