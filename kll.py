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
        c = np.vstack((self.c, new_seed(self.X, 1, beta)))
        # line 3 algorithm 3
        alpha = np.array([np.inf] * self.n).reshape((-1, 1))
        k = 1
        kp = 0
        # line 4,5,6 algorithm 3
        for r in range(self.R):
            for i in range(self.n):
                for j in range(kp, k):
                    # line 7 algorithm 3
                    alpha[i] = np.minimum(alpha[i], distance(self.X[i], c[j]))

            # line 8,9 algorithm 3
            kp = k
            t = self.sample_weight * (alpha**2)
            z = np.sum(t)
            # line 10,11,12 algorithm 3
            for i in range(self.n):
                p = self.L * self.sample_weight[i] * (alpha[i] ** 2) / z
                if p > np.random.rand(1)[0]:
                    k += 1
                    c = np.vstack((c, self.X[i]))
                    alpha[i] = 0

        # line 13 algorithm 3
        wp = np.empty((0, 1))
        for i in range(c.shape[0]):
            s = 0
            for j in range(self.n):
                wj = self.sample_weight[j]
                indic = 1 if distance(self.X[j], c[i]) == alpha[j] else 0
                s += wj * indic

            wp = np.vstack((wp, s))
        self.c = c
        # line 14 algorithm 3
        kpp_ = KPP(Dataset(np.array(c)))

        return kpp_.fit(self.K, wp)
