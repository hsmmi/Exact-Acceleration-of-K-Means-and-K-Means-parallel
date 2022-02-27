from kpp import KPP
from nearest_neighbor_search import NNS
from tools import distance, execution_time, new_seed
import numpy as np
from dataset import Dataset


class AKLL:
    def __init__(self, dataset: Dataset) -> None:
        self.X = dataset.sample
        self.n = dataset.number_of_sample
        self.d = dataset.number_of_feature
        self.w = None
        self.K = None
        self.R = None
        self.L = None
        self.c = np.empty((0, self.d))

    @execution_time
    def fit(
        self,
        K: int,
        R: int = 5,
        L: int = None,
        w: np.ndarray = None,
    ) -> np.ndarray:
        """Find K initial seeds for k-means algorithm
        We'll find seed with Accelerated K-Means|| methon

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
            self.n.shape[0] == self.w.shape[0]
        ), "size weights should be nx1(number of sample"

        # Line 1 algorithm 5
        beta = self.w / np.sum(self.w)
        # Line 2 algorithm 5
        self.c = np.vstack((self.c, new_seed(self.X, 1, beta)))
        # Line 3 algorithm 5
        # In each iteration new centers are in c(k_pre, k]
        # k always point to last center
        alpha = np.full((self.n, 1), np.inf)
        k_pre, k = -1, 0
        # Line 4 algorithm 5
        for r in range(R):
            if k - (k_pre + 1) + 1 > 0:  # we create center in previous round
                # Line 5 algorithm 5
                C = NNS(
                    self.c[k_pre + 1 : k + 1],
                    np.arange(start=k_pre + 1, stop=k + 1),
                )
                # Line 6 - 9 algorithm 5
                for i in range(self.n):
                    dis, j = C.nearest_in_range(self.X[i], alpha[i])
                    if j >= 0:
                        alpha[i] = dis
            # Line 10 algorithm 5
            k_pre = k
            Z = np.sum(self.w * (alpha**2))
            # Line 11 algorithm 5
            for i in range(self.n):
                p = min(1, self.L * self.w[i] * (alpha[i] ** 2) / Z)
                # Line 12 algorithm 5
                if p > np.random.rand(1)[0]:
                    # X[i] become new center
                    # Line 14 algorithm 5
                    k = k + 1
                    self.c = np.vstack((self.c, self.X[i]))
                    alpha[i] = 0

        # Line 14 algorithm 5
        w_p = np.empty((0, 1))
        dist_center_point = distance(self.c, self.X)
        for i in range(self.c.shape[0]):
            sum_w_p_i = 0
            for j in range(self.n):
                if dist_center_point[i, j] == alpha[j]:
                    sum_w_p_i += self.w[j]
            w_p = np.vstack((w_p, sum_w_p_i))

        # Line 15 algorithm 5
        dataset_center = Dataset(self.c)
        kpp = KPP(dataset_center)
        return kpp.fit(self.K, w_p)
