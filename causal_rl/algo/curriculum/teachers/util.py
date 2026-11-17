# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/teachers/util.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
import numpy as np
import nadaraya_watson as na


class Buffer:
    """`n_elements` parallel lists holding at most `max_buffer_size` entries (oldest dropped first).
    `read_buffer()` returns the lists and empties the buffer."""

    def __init__(self, n_elements, max_buffer_size):
        self.max_buffer_size = max_buffer_size
        self.buffers = [list() for i in range(0, n_elements)]

    def update_buffer(self, datas):
        for buffer, data in zip(self.buffers, datas):
            buffer.append(data)

        while len(self.buffers[0]) > self.max_buffer_size:
            for buffer in self.buffers:
                del buffer[0]

    def read_buffer(self):
        res = tuple(self.buffers)
        self.buffers = [list() for _ in self.buffers]
        return res

    def __len__(self):
        return len(self.buffers[0])


class NadarayaWatson:
    """Nadaraya-Watson kernel regression of episode returns over contexts (C++ backend)."""

    def __init__(self, contexts, returns, lengthscale, n_threads=5, n_max=None, radius_scale=3.):
        self.model = na.NadarayaWatson(contexts, returns, n_threads=n_threads)
        self.lengthscale = lengthscale

        if n_max is None:
            self.n_max = int(0.5 * contexts.shape[0])
        else:
            self.n_max = n_max

        self.radius_scale = radius_scale

    def predict_individual(self, x):
        return np.reshape(self.model.predict(np.reshape(x, (-1, x.shape[-1])), self.lengthscale, n_max=self.n_max,
                                             radius_scale=self.radius_scale), x.shape[:-1])
