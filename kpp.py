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
    def fit(self, K: int, w: np.ndarray = None) -> np.ndarray:
        """Find K initial seeds for k-means algorithm
        We'll find seed with Accelerated K-Means++ methon

        Args:
            K (int): number of cluseter
            w (ndarray): nx1 ndarray for weights of n sample (default 1)

        Returns:
            Initial K seed(s)
        """
        assert K < self.n, "number of cluster is greater than number of sample"
        self.K = K
        self.m = np.empty((0, self.d))
        if w is None:
            self.w = np.ones((self.n, 1))
        else:
            self.w = w.reshape((-1, 1))
        assert (
            self.n.shape[0] == self.w.shape[0]
        ), "size weights should be nx1(number of sample"

        # line 1 algorithm 1
        beta = self.w / np.sum(self.w)
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
            t = self.w * (alpha**2)
            beta = t / np.sum(t)
            # line 9 algorithm 1
            k += 1
            # line 10 algorithm 1
            m = np.vstack((m, new_seed(self.X, 1, beta)))
        # line 11 algorithm 1
        self.m = m
        return np.array(m)
