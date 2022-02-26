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

    @execution_time
    def fit(
        self,
        number_of_cluster: int,
        rounds: int = 5,
        oversampling_factor: int = 1,
        sample_weight: np.ndarray = None,
    ) -> np.ndarray:

        self.c = np.empty((0, self.d))
        self.K = number_of_cluster
        self.R = rounds
        self.L = oversampling_factor

        assert self.R > 0 and self.L > 0, "invalid function parameter"
        assert (
            self.n > self.K and self.K > 0
        ), "number of cluster should be in range [1,n)"

        if sample_weight is not None:
            self.sample_weight = np.array(sample_weight)
        else:
            self.sample_weight = np.ones(self.n)

        self.sample_weight = self.sample_weight.reshape((-1, 1))
        # line 1 algorithm 3
        beta = self.sample_weight / np.sum(self.sample_weight)
        # line 2 algorithm 3
        self.c = np.vstack((self.c, new_seed(self.X, 1, beta)))
        # line 3 algorithm 3
        alpha = np.full((self.n, 1), np.inf)
        k = 0
        k_p = -1
        # line 4,5,6 algorithm 3
        for r in range(self.R):
            for i in range(self.n):
                for j in range(k_p + 1, k + 1):
                    # line 7 algorithm 3
                    alpha[i] = min(alpha[i], distance(self.X[i], self.c[j]))

            # line 8,9 algorithm 3
            k_p = k
            Z = np.sum(self.sample_weight * (alpha**2))
            # line 10,11,12 algorithm 3
            for i in range(self.n):
                p = min(
                    1, self.L * self.sample_weight[i] * (alpha[i] ** 2) / Z
                )
                if p > np.random.rand(1)[0]:
                    k += 1
                    self.c = np.vstack((self.c, self.X[i]))
                    alpha[i] = 0

        # line 13 algorithm 3
        w_p = np.empty((0, 1))

        dist_center_point = distance(self.c, self.X)
        for i in range(self.c.shape[0]):
            sum_w_p_i = 0
            for j in range(self.n):
                if dist_center_point[i, j] == alpha[j]:
                    sum_w_p_i += self.sample_weight[j]
            w_p = np.vstack((w_p, sum_w_p_i))

            w_p = np.vstack((w_p, sum_w_p_i))

        # line 14 algorithm 3
        kpp_ = KPP(Dataset(np.array(self.c)))

        return kpp_.fit(self.K, w_p)
