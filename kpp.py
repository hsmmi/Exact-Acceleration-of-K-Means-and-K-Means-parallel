import numpy as np
from dataset import Dataset
from tools import distance, new_seed, execution_time


# algorithm 1 K-Means++
class KPP:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.n = dataset.number_of_sample
        self.d = dataset.number_of_feature
        self.X = dataset.sample
        self.m = None
        self.K = None

    @execution_time
    def fit(
        self, number_of_cluster: int, sample_weight: np.ndarray = None
    ) -> np.ndarray:

        self.m = np.empty((0, self.d))
        self.K = number_of_cluster
        assert (
            self.n > self.K and self.K > 0
        ), "number of cluster should be in range [1,n)"

        if sample_weight is not None:
            self.sample_weight = np.array(sample_weight)
        else:
            self.sample_weight = np.ones(self.n)
        self.sample_weight = self.sample_weight.reshape((-1, 1))
        # line 1 algorithm 1
        beta = self.sample_weight / np.sum(self.sample_weight)
        # line 2 algorithm 1
        m = np.vstack((self.m, new_seed(self.X, 1, beta)))
        # line 3 algorithm 1
        alpha = np.array([np.inf] * self.n).reshape((-1, 1))
        k = 1
        # line 4 algorithm 1
        while k < self.K:
            # line 5,6 algorithm 1
            alpha = np.minimum(alpha, distance(self.X, m[k - 1]))
            # line 7,8 algorithm 1
            t = self.sample_weight * (alpha**2)
            beta = t / np.sum(t)
            # line 9 algorithm 1
            k += 1
            # line 10 algorithm 1
            m = np.vstack((m, new_seed(self.X, 1, beta)))
        # line 11 algorithm 1
        self.m = m
        return np.array(m)
