# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/teachers/goal_gan/generator.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
# The following code was obtained from https://github.com/florensacc/rllab-curriculum
# and only slightly modified to fit the project structure. The original license of
# the software is the following:

# The MIT License (MIT)

# Copyright (c) 2016 rllab contributors

# rllab uses a shared copyright model: each contributor holds copyright over
# their contributions to rllab. The project versioning records all such
# contribution and copyright details.
# By contributing to the rllab repository through pull-request, comment,
# or otherwise, the contributor releases their content to the license and
# copyright terms herein.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy.spatial.distance import cdist
from causal_rl.algo.curriculum.teachers.goal_gan.gan import FCGAN


def sample_matrix_row(M, size, replace=False):
    if size > M.shape[0]:
        return M
    if replace:
        indices = np.random.randint(0, M.shape[0], size)
    else:
        indices = np.random.choice(M.shape[0], size, replace=replace)
    return M[indices, :]


class StateGAN(object):
    """A GAN for generating states (goals), wrapping FCGAN with (un-)normalization."""

    def __init__(self, state_size, evaluater_size, state_noise_level, state_range, state_center=None, **kwargs):
        self.gan = FCGAN(
            generator_output_size=state_size,
            discriminator_output_size=evaluater_size,
            **kwargs
        )
        self.state_size = state_size
        self.evaluater_size = evaluater_size
        self.state_center = np.array(state_center) if state_center is not None else np.zeros(state_size)
        self.state_range = state_range
        self.state_bounds = np.vstack(
            [-self.state_range * np.ones(self.state_size), self.state_range * np.ones(self.state_size)])
        self.state_noise_level = state_noise_level

    def pretrain(self, states, outer_iters=500):
        """
        Pretrain the state GAN to match the distribution of given states.
        :param states: the state distribution to match
        :param outer_iters: of the GAN
        """
        labels = np.ones((states.shape[0], self.evaluater_size))  # all state same label --> uniform
        return self.train(states, labels, outer_iters)

    def _add_noise_to_states(self, states):
        noise = np.random.randn(*states.shape) * self.state_noise_level
        states += noise
        return np.clip(states, self.state_center + self.state_bounds[0], self.state_center + self.state_bounds[1])

    def sample_states(self, size):  # un-normalizes the states
        normalized_states, noise = self.gan.sample_generator(size)
        states = self.state_center + normalized_states * (self.state_bounds[1] - self.state_bounds[0])
        return states, noise

    def sample_states_with_noise(self, size):
        states, noise = self.sample_states(size)
        states = self._add_noise_to_states(states)
        return states, noise

    def train(self, states, labels, outer_iters):
        normalized_states = (states - self.state_center) / (self.state_bounds[1] - self.state_bounds[0])
        return self.gan.train(normalized_states, labels, outer_iters)


class StateCollection(object):
    """ A collection of states, with minimum distance threshold for new states. """

    def __init__(self, distance_threshold=None):
        self.distance_threshold = distance_threshold
        self.state_list = []

    @property
    def size(self):
        return len(self.state_list)

    def sample(self, size, replace=False, replay_noise=0):
        states = sample_matrix_row(np.array(self.state_list), size, replace)
        if len(states) != 0:
            states += replay_noise[None, :] * np.random.randn(*states.shape)
        return states

    def append(self, states):
        if len(states) > 0:
            states = np.array(states)
            if self.distance_threshold is not None and self.distance_threshold > 0:
                states = self._process_states(states)
            states = self._select_states(states)
            self.state_list.extend(states.tolist())
            return states

    def _select_states(self, states):
        """Keep only the states that are at more than distance_threshold from the collection."""
        selected_states = states
        if self.distance_threshold is not None and self.distance_threshold > 0:
            if len(self.state_list) > 0:
                dists = cdist(np.array(self.state_list), selected_states)
                indices = np.amin(dists, axis=0) > self.distance_threshold
                selected_states = selected_states[indices, :]
        return selected_states

    def _process_states(self, states):
        """Keep only the states that are at more than distance_threshold from each other."""
        states = np.array(states)
        results = [states[0]]
        for state in states[1:]:
            if np.amin(cdist(results, state.reshape(1, -1))) > self.distance_threshold:
                results.append(state)
        return np.array(results)
