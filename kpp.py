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
        self.w = None

    @execution_time
    def fit(self, K: int, w: np.ndarray = None) -> np.ndarray:
        """Find K initial seeds for k-means algorithm
        We'll find seed with K-Means++ method

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
            self.n == self.w.shape[0]
        ), "size weights should be nx1(number of sample)"

        # Line 1 algorithm 1
        # Probability to select next center
        beta = self.w / np.sum(self.w)
        # Line 2 algorithm 1
        # Select first center with probabiliry beta
        m = np.vstack((self.m, new_seed(self.X, 1, beta)))
        # Line 3 algorithm 1
        # Distance to closest mean
        alpha = np.full((self.n, 1), np.inf)
        # k always point to last center
        k = 1
        # Line 4 algorithm 1
        while k < self.K:
            # Line 5,6 algorithm 1
            # Update alpha
            alpha = np.minimum(alpha, distance(self.X, m[k - 1]))
            # Line 7,8 algorithm 1
            # Update probability to select new center
            t = self.w * (alpha**2)
            beta = t / np.sum(t)
            # Line 9 algorithm 1
            k += 1
            # Line 10 algorithm 1
            # Select new center with probabiliry beta
            m = np.vstack((m, new_seed(self.X, 1, beta)))
        # Line 11 algorithm 1
        self.m = m
        return np.array(m)
