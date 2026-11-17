# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/teachers/alp_gmm/alp_gmm_wrapper.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
from causal_rl.algo.curriculum.base_wrapper import BaseWrapper


class ALPGMMWrapper(BaseWrapper):

    def done_callback(self, step, context, discounted_reward):
        self.teacher.update(context, discounted_reward)
