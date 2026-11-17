# From CausalAILab/Causally-Aligned-Curriculum-Learning @ f52fcea, deep_sprl/teachers/alp_gmm/dataset.py
# (MIT License, (c) 2026 Mingxuan Li). Only the import paths are changed.
# Modified version of the dataset from https://github.com/flowersteam/explauto
# Implements a buffered k-nearest-neighbor lookup using a kd-tree
import numpy as np
import scipy.spatial

DATA_Y = 1  # index of the output data in Dataset.data (0 = input data)


class Dataset(object):
    """Hold observations and provide nearest neighbors facilities"""

    def __init__(self, dim_x, dim_y, lateness=0, max_size=None):
        """
            :arg dim_x:  the dimension of the input vectors
            :arg dim_y:  the dimension of the output vectors
        """
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.lateness = lateness
        self.max_size = max_size

        self.reset()

    def reset(self):
        """Reset the dataset to zero elements."""
        self.data = [[], []]
        self.size = 0
        self.kdtree = [None, None]  # KDTreeX, KDTreeY
        self.nn_ready = [False, False]  # if True, the tree is up-to-date.
        self.late_points = 0

    def add_xy(self, x, y=None):
        self.data[0].append(x)
        if self.dim_y > 0:
            self.data[1].append(y)
        self.size += 1
        if self.late_points == self.lateness:
            self.nn_ready = [False, False]
            self.late_points = 0
        else:
            self.late_points += 1
        # Reduce data size
        if self.max_size and self.size > self.max_size:
            n = self.size - self.max_size
            del self.data[0][:n]
            del self.data[1][:n]
            self.size = self.max_size

    def get_x(self, index):
        return self.data[0][index]

    def __len__(self):
        return self.size

    def _nn(self, side, v, k=1, radius=np.inf, eps=0.0, p=2):
        """Compute the k nearest neighbors of v in the observed data,
        :arg side  0: search among input data, DATA_Y: search among output data.
        @return  distance and indexes of found nearest neighbors.
        """
        self._build_tree(side)
        dists, idxes = self.kdtree[side].query(v, k=k, distance_upper_bound=radius,
                                               eps=eps, p=p)
        if k == 1:
            dists, idxes = np.array([dists]), [idxes]
        return dists, idxes

    def _build_tree(self, side):
        """Build the KDTree for the observed data
        :arg side  0: build input data tree, DATA_Y: build output data tree.
        """
        if not self.nn_ready[side]:
            self.kdtree[side] = scipy.spatial.cKDTree(self.data[side], compact_nodes=False,
                                                      balanced_tree=False)  # Those options are required with scipy >= 0.16
            self.nn_ready[side] = True


class BufferedDataset(Dataset):
    """Add a buffer of a few points to avoid recomputing the kdtree at each addition"""

    def __init__(self, dim_x, dim_y, buffer_size=200, lateness=5, max_size=None):
        """
            :arg dim_x:  the dimension of the input vectors
            :arg dim_y:  the dimension of the output vectors
        """

        self.buffer_size = buffer_size
        self.lateness = lateness
        self.buffer = Dataset(dim_x, dim_y, lateness=self.lateness)

        Dataset.__init__(self, dim_x, dim_y, lateness=0)
        self.max_size = max_size

    def reset(self):
        self.buffer.reset()
        Dataset.reset(self)

    def add_xy(self, x, y=None):
        if self.buffer.size < self.buffer_size:
            self.buffer.add_xy(x, y)
        else:
            self.data[0] = self.data[0] + self.buffer.data[0]
            if self.dim_y > 0:
                self.data[1] = self.data[1] + self.buffer.data[1]
            self.size += self.buffer.size
            self.buffer = Dataset(self.dim_x, self.dim_y, lateness=self.lateness)
            self.nn_ready = [False, False]
            self.buffer.add_xy(x, y)

            # Reduce data size
            if self.max_size and self.size > self.max_size:
                n = self.size - self.max_size
                del self.data[0][:n]
                del self.data[1][:n]
                self.size = self.max_size

    def get_x(self, index):
        if index >= self.size:
            return self.buffer.data[0][index - self.size]
        else:
            return self.data[0][index]

    def __len__(self):
        return self.size + self.buffer.size

    def nn_y(self, y, k=1, radius=np.inf, eps=0.0, p=2):
        """Find the k nearest neighbors of y in the observed output data
        @return  distance and indexes of found nearest neighbors.
        """
        assert len(y) == self.dim_y
        k_y = min(k, self.__len__())
        return self._nn(DATA_Y, y, k=k_y, radius=radius, eps=eps, p=p)

    def _nn(self, side, v, k=1, radius=np.inf, eps=0.0, p=2):
        """Compute the nearest neighbor of v in the observed data (only k=1 is supported),
        :arg side  0: search among input data, DATA_Y: search among output data.
        @return  distance and indexes of found nearest neighbors.
        """
        if self.size > 0:
            dists, idxes = Dataset._nn(self, side, v, k, radius, eps, p)
        else:
            return self.buffer._nn(side, v, k, radius, eps, p)
        if self.buffer.size > 0:
            buffer_dists, buffer_idxes = self.buffer._nn(side, v, k, radius, eps, p)
            buffer_idxes = [i + self.size for i in buffer_idxes]
            if dists[0] <= buffer_dists:
                return dists, idxes
            else:
                return buffer_dists, buffer_idxes
        else:
            return dists, idxes
