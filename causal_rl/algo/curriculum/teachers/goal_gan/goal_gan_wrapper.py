# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/teachers/goal_gan/goal_gan_wrapper.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
from causal_rl.algo.curriculum.base_wrapper import BaseWrapper


class GoalGANWrapper(BaseWrapper):

    def done_callback(self, step, context, discounted_reward):
        # GoalGAN is trained on binary success labels reported by the env in info['success']
        self.teacher.update(context, float(step[4]["success"]))
