# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/teachers/alp_gmm/alp_gmm.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
import numpy as np
from sklearn.mixture import GaussianMixture as GMM

from causal_rl.algo.curriculum.teachers.abstract_teacher import AbstractTeacher
from causal_rl.algo.curriculum.teachers.alp_gmm.dataset import BufferedDataset


def proportional_choice(v, eps=0.):
    if np.sum(v) == 0 or np.random.rand() < eps:
        return np.random.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(np.random.multinomial(1, probas) == 1)[0][0]


class EmpiricalALPComputer:
    """Absolute learning progress (ALP) via a buffered kd-tree nearest-neighbor lookup."""

    def __init__(self, task_size, max_size=None, buffer_size=500):
        self.alp_knn = BufferedDataset(1, task_size, buffer_size=buffer_size, lateness=0, max_size=max_size)

    def compute_alp(self, task, reward):
        alp = 0
        if len(self.alp_knn) > 5:
            # ALP = |reward - reward of the closest previous task|
            _, idx = self.alp_knn.nn_y(task)
            closest_previous_task_reward = self.alp_knn.get_x(idx[0])
            lp = reward - closest_previous_task_reward
            alp = np.abs(lp)

        self.alp_knn.add_xy(reward, task)
        return alp


class ALPGMM(AbstractTeacher):
    """
    Absolute Learning Progress - Gaussian Mixture Model teacher (Portelas et al., 2019).
    mins / maxs define the task space boundaries.
    """

    def __init__(self, mins, maxs, fit_rate=250, random_task_ratio=0.2, max_size=None, seed=None):
        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42, 424242)
        np.random.seed(self.seed)

        # Task space boundaries
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)

        # Range of number of Gaussians to try when fitting the GMM
        self.potential_ks = np.arange(2, 11, 1)
        # Number of episodes between two fits of the GMM (also the number of bootstrapping episodes)
        self.fit_rate = fit_rate
        self.nb_random = self.fit_rate

        # Ratio of randomly sampled tasks VS tasks sampled using the GMM
        self.random_task_ratio = random_task_ratio

        # max_size: maximal number of episodes to account for when computing ALP
        self.alp_computer = EmpiricalALPComputer(len(mins), max_size=max_size, buffer_size=500)

        self.tasks = []
        self.alps = []
        self.tasks_alps = []

        self.potential_gmms = [self.init_gmm(k) for k in self.potential_ks]

    def init_gmm(self, nb_gaussians):
        return GMM(n_components=nb_gaussians, covariance_type='full', random_state=self.seed,
                   warm_start=False, n_init=1)

    def update(self, task, reward):
        self.tasks.append(task)

        # Compute the corresponding ALP and concatenate it to the task vector
        self.alps.append(self.alp_computer.compute_alp(task, reward))
        self.tasks_alps.append(np.array(task.tolist() + [self.alps[-1]]))

        if len(self.tasks) >= self.nb_random:  # If initial bootstrapping is done
            if (len(self.tasks) % self.fit_rate) == 0:  # Time to fit
                # Fit GMMs with varying number of Gaussians on the last <fit_rate> (task, ALP) pairs
                cur_tasks_alps = np.array(self.tasks_alps[-self.fit_rate:])
                self.potential_gmms = [g.fit(cur_tasks_alps) for g in self.potential_gmms]

                # Keep the best GMM w.r.t. the Akaike Information Criterion
                fitnesses = [m.aic(cur_tasks_alps) for m in self.potential_gmms]
                self.gmm = self.potential_gmms[np.argmin(fitnesses)]

    def sample(self):
        if (len(self.tasks) < self.nb_random) or (np.random.random() < self.random_task_ratio):
            # Random task sampling
            new_task = np.random.uniform(self.mins, self.maxs).astype(np.float32)
        else:
            # ALP-based task sampling: pick a Gaussian proportionally to its mean ALP
            alp_means = [pos[-1] for pos in self.gmm.means_]
            idx = proportional_choice(alp_means, eps=0.0)

            # Sample a task from the Gaussian, removing the ALP dimension
            new_task = np.random.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx])[:-1]
            new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)

        return new_task
