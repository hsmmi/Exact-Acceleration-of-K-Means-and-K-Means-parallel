from kpp import kpp
from nearest_neighbor_search import NNS
from tools import distance, new_seed
import numpy as np
from dataset import Dataset


class AKPP:
    def __init__(self, dataset: Dataset) -> None:
        self.X = dataset.sample
        self.n = dataset.number_of_sample
        self.d = dataset.number_of_feature
        self.w = None
        self.K = None
        self.R = None
        self.L = None
        self.c = np.empty((0, self.d))

    def fit(
        self,
        number_of_cluster: int,
        R: int,
        L: int,
        sample_weight: np.ndarray = None,
    ):
        assert (
            number_of_cluster < self.n
        ), "number of cluster is greater than number of sample"
        self.K = number_of_cluster
        assert R > 0, "Number of round(R) shoud be greater than 0"
        assert L > 0, "Number of round(L) shoud be greater than 0"

        self.c = np.empty((0, self.d))

        if sample_weight is None:
            self.w = np.ones((self.n, 1))
        else:
            self.w = sample_weight.reshape((-1, 1))

        # line 1 algorithm 5
        beta = self.w / np.sum(self.w)
        # line 2 algorithm 5
        self.c = np.vstack((self.c, new_seed(self.X, 1, beta)))
        # line 3 algorithm 2
        # In each iteration new centers are in c(k_pre, k]
        # k always point to last center
        alpha = np.full((self.n, 1), np.inf)
        k_pre, k = -1, 0
        # line 4 algorithm 2
        for r in range(R):
            # line 5 algorithm 2
            C = NNS(
                self.c[k_pre + 1 : k + 1],
                np.arange(start=k_pre + 1, stop=k + 1),
            )
            # line 6 - 9 algorithm 2
            for i in range(self.n):
                dis, j = C.nearest_in_range(self.X[i], alpha[i])
                if j >= 0:
                    alpha[i] = dis
            # line 10 algorithm 2
            k_pre = k
            Z = np.sum(self.w * (alpha**2))
            # line 11 algorithm 2
            for i in range(self.n):
                p = min(1, self.L * self.w[i] * (alpha[i] ** 2) / Z)
                # line 12 algorithm 2
                if p > np.random.rand(1)[0]:
                    # X[i] become new center
                    # line 14 algorithm 2
                    k = k + 1
                    self.c = np.vstack((self.c, self.X[i]))
                    alpha[i] = 0

        # line 14 algorithm 2
        w_p = np.empty((0, 1))
        dist_center_point = distance(self.c, self.X)
        for i in range(self.c.shape[0]):
            sum_w_p_i = 0
            for j in range(self.n):
                if dist_center_point[i, j]:
                    sum_w_p_i += self.w[j]
            w_p = np.vstack((w_p, sum_w_p_i))

        # line 15 algorithm 2
        return kpp(self.K, self.c, w_p)