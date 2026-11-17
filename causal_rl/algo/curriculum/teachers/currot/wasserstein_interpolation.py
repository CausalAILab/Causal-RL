# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/teachers/spl/wasserstein_interpolation.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
import numpy as np
from scipy.optimize import linear_sum_assignment
from causal_rl.algo.curriculum.teachers.currot.sliced_wasserstein import sliced_wasserstein


class SamplingWassersteinInterpolation:
    """
    Particle-based interpolation from the current context distribution towards the target
    distribution under the constraint that the predicted performance stays above `perf_lb`
    (the CurrOT update). In each update, particles with a predicted performance below `perf_lb`
    are first replaced by contexts from the success buffer (minimum-distance assignment); every
    particle then moves at most `epsilon` from that starting point.
    """

    def __init__(self, init_samples, target_sampler, perf_lb, epsilon, bounds):
        self.current_samples = init_samples
        self.n_samples, self.dim = self.current_samples.shape
        self.target_sampler = target_sampler
        self.bounds = bounds
        self.perf_lb = perf_lb
        self.epsilon = epsilon

    def sample_ball(self, targets, samples=None, half_ball=None, n=100):
        if samples is None:
            samples = self.current_samples

        # Taken from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        # Method 20
        direction = np.random.normal(0, 1, (n, self.dim))
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        r = np.power(np.random.uniform(size=(n, 1)), 1. / self.dim)

        # We only consider samples that decrease the distance objective (i.e. are aligned with the direction)
        noise = r * (direction / norm)
        dirs = targets - samples
        dir_norms = np.einsum("ij,ij->i", dirs, dirs)
        noise_projections = np.einsum("ij,kj->ik", dirs / dir_norms[:, None], noise)

        projected_noise = np.where((noise_projections > 0)[..., None], noise[None, ...],
                                   noise[None, ...] - 2 * noise_projections[..., None] * dirs[:, None, :])
        if half_ball is not None:
            projected_noise[~half_ball] = noise

        scales = np.minimum(self.epsilon, np.sqrt(dir_norms))[:, None, None]
        return np.clip(samples[..., None, :] + scales * projected_noise, self.bounds[0], self.bounds[1])

    def ensure_successful_initial(self, model, init_samples, success_samples):
        performance_reached = model.predict_individual(init_samples) >= self.perf_lb
        replacement_mask = ~performance_reached
        n_replacements = np.sum(replacement_mask)
        if n_replacements > 0:
            valid_successes = model.predict_individual(success_samples) >= self.perf_lb
            n_valid = np.sum(valid_successes)
            if n_valid >= n_replacements:
                # In this case we only allow selection from the valid samples
                success_samples = success_samples[valid_successes, :]
                valid_successes = np.ones(success_samples.shape[0], dtype=bool)

            dists = np.sum(np.square(success_samples[None, :, :] - init_samples[replacement_mask, None, :]), axis=-1)
            success_assignment = linear_sum_assignment(dists, maximize=False)
            init_samples[replacement_mask, :] = success_samples[success_assignment[1], :]
            performance_reached[replacement_mask] = valid_successes[success_assignment[1]]

        return init_samples, performance_reached

    def update_distribution(self, model, success_samples):
        init_samples, performance_reached = self.ensure_successful_initial(model, self.current_samples.copy(),
                                                                           success_samples)
        target_samples = self.target_sampler(self.n_samples)
        movements = sliced_wasserstein(init_samples, target_samples, grad=True)[1]
        target_samples = init_samples + movements
        particles = self.sample_ball(target_samples, samples=init_samples, half_ball=performance_reached)

        distances = np.linalg.norm(particles - target_samples[:, None, :], axis=-1)
        performances = model.predict_individual(particles)

        mask = performances > self.perf_lb
        solution_possible = np.any(mask, axis=-1)
        distances[~mask] = np.inf
        opt_idxs = np.where(solution_possible, np.argmin(distances, axis=-1), np.argmax(performances, axis=-1))
        new_samples = particles[np.arange(0, self.n_samples), opt_idxs]

        self.current_samples = new_samples
