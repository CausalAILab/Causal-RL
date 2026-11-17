# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/teachers/dummy_teachers.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
import numpy as np
from causal_rl.algo.curriculum.teachers.abstract_teacher import AbstractTeacher


class DistributionSampler(AbstractTeacher):
    """Samples contexts from `sample_func` restricted to [lb, ub] (used by the default curriculum)."""

    def __init__(self, sample_func, lb, ub):
        self.sample_func = sample_func
        self.lb = lb
        self.ub = ub
        self.dim = self.sample_func(n=1).shape[1]

        # These samples are never used: they are drawn only to reproduce the original code's
        # global RNG consumption (it estimated the mean/covariance of the restricted distribution).
        # Note: in the first iteration `samples_not_ok` is an int8 array, i.e. this is integer (not
        # boolean) indexing.
        n_samples = int(1e6)
        samples_not_ok = np.ones(n_samples, dtype=np.int8)
        samples = np.zeros((n_samples, self.dim))
        while np.any(samples_not_ok):
            n_new_samples = np.sum(samples_not_ok)
            samples[samples_not_ok] = self.sample_func(n=n_new_samples)
            samples_not_ok = np.logical_or(np.any(samples < lb, axis=-1),
                                           np.any(samples > ub, axis=-1))

    def sample(self):
        ok = False
        while not ok:
            sample = self.sample_func(n=1)[0, :]
            ok = np.all(sample >= self.lb) and np.all(sample <= self.ub)
        return sample


class UniformSampler(AbstractTeacher):
    """Samples contexts uniformly from the box [lower_bound, upper_bound]."""

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sample(self):
        norm_sample = np.random.uniform(low=-1, high=1, size=self.lower_bound.shape)
        return self._scale_context(norm_sample)

    def _scale_context(self, context):
        b = 0.5 * (self.upper_bound + self.lower_bound)
        m = 0.5 * (self.upper_bound - self.lower_bound)
        return m * context + b
