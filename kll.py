import numpy as np
from dataset import Dataset
from tools import distance, new_seed, execution_time
from kpp import KPP


# algorithm 3 K-Means ||
class KLL:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.n = dataset.number_of_sample
        self.d = dataset.number_of_feature
        self.X = dataset.sample
        self.c = None
        self.K = None
        self.R = None
        self.L = None
        self.w = None

    @execution_time
    def fit(
        self,
        K: int,
        R: int = 5,
        L: int = None,
        w: np.ndarray = None,
    ) -> np.ndarray:

        """Find K initial seeds for k-means algorithm
        We'll find seed with K-Means|| methon

        Assert:
            K < R x L

        Args:
            K (int): number of cluseter
            R (int): number of round(s) (default = 5)
            L (int): size of oversampling (default = 2xK)
            w (ndarray): nx1 ndarray for weights of n sample (default 1)

        Returns:
            Initial K seed(s)
        """
        assert (
            K < self.n
        ), "number of cluster(K) is greater than number of sample"
        self.K = K
        assert R > 0, "Number of round(R) shoud be greater than 0"
        self.R = R
        if L is None:
            L = 2 * self.K
        assert L > 0, "Number of round(L) shoud be greater than 0"
        self.L = L
        assert R * L > K, "RxL shoulf be greater than number of cluster"

        self.c = np.empty((0, self.d))

        if w is None:
            self.w = np.ones((self.n, 1))
        else:
            self.w = w.reshape((-1, 1))
        assert (
            self.n == self.w.shape[0]
        ), "size weights should be nx1(number of sample"

        self.w = self.w.reshape((-1, 1))
        # Line 1 algorithm 3
        beta = self.w / np.sum(self.w)
        # Line 2 algorithm 3
        self.c = np.vstack((self.c, new_seed(self.X, 1, beta)))
        # Line 3 algorithm 3
        alpha = np.full((self.n, 1), np.inf)
        k = 0
        k_p = -1
        # Line 4,5,6 algorithm 3
        for r in range(self.R):
            for i in range(self.n):
                for j in range(k_p + 1, k + 1):
                    # Line 7 algorithm 3
                    alpha[i] = min(alpha[i], distance(self.X[i], self.c[j]))

            # Line 8,9 algorithm 3
            k_p = k
            Z = np.sum(self.w * (alpha**2))
            # Line 10,11,12 algorithm 3
            for i in range(self.n):
                p = min(1, self.L * self.w[i] * (alpha[i] ** 2) / Z)
                if p > np.random.rand(1)[0]:
                    k += 1
                    self.c = np.vstack((self.c, self.X[i]))
                    alpha[i] = 0

        # Line 13 algorithm 3
        w_p = np.empty((0, 1))

        dist_center_point = distance(self.c, self.X)
        for i in range(self.c.shape[0]):
            sum_w_p_i = 0
            for j in range(self.n):
                if dist_center_point[i, j] == alpha[j]:
                    sum_w_p_i += self.w[j]
            w_p = np.vstack((w_p, sum_w_p_i))

        # Line 14 algorithm 3
        kpp_ = KPP(Dataset(self.c))

        return kpp_.fit(self.K, w_p)
