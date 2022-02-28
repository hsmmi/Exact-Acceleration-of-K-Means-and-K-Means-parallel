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
        # Probability to select next center
        beta = self.w / np.sum(self.w)
        # Line 2 algorithm 3
        # Select first center with probabiliry beta
        self.c = np.vstack((self.c, new_seed(self.X, 1, beta)))
        # Line 3 algorithm 3
        # alpha: Distance to closest mean
        alpha = np.full((self.n, 1), np.inf)
        # In each iteration new centers are in c(k_pre, k]
        # k always point to last center
        k_p, k = -1, 0
        # Line 4,5,6 algorithm 3
        for r in range(self.R):
            for i in range(self.n):
                for j in range(k_p + 1, k + 1):
                    # Line 7 algorithm 3
                    # Update alpha i
                    alpha[i] = min(alpha[i], distance(self.X[i], self.c[j]))

            # Line 8,9 algorithm 3
            # Now we want to create new centers
            # So now our next last previous center
            # is k which is current last center
            k_p = k
            # Normalize the probability of selection
            Z = np.sum(self.w * (alpha**2))
            # Line 10 - 12 algorithm 3
            for i in range(self.n):
                # Expect to select L new centers
                p = min(1, self.L * self.w[i] * (alpha[i] ** 2) / Z)
                if p > np.random.rand(1)[0]:
                    # Is point selected as center?
                    k += 1
                    self.c = np.vstack((self.c, self.X[i]))
                    # Now distance this point to closest center is zere(itself)
                    # As alpha[i] become zero the probability of selecting this
                    # point in next round gets zero so each point select
                    alpha[i] = 0

        # Line 13 algorithm 3
        # Now we have aound 2xKxR centers and should select K of them
        # as initial seeds and we do it with K-means++ algorithm
        # as set weights of each center to sum of wight of points
        # that are assign to this center and store it in w_p
        w_p = np.empty((0, 1))
        # To find if a point assign to a center we should find distances
        # between all pairs of center and point then if distance of point[j]
        # to the center[i] be equal to alpha[j] then we know that the closest
        # center to point[j] is center[i]
        dist_center_point = distance(self.c, self.X)
        # For each center
        for i in range(self.c.shape[0]):
            # Initial the weights for this center to zero
            sum_w_p_i = 0
            # For each point
            for j in range(self.n):
                # If distance of closest center to our point(alpha[j]) is
                # equal to distance this center to our point the this center
                # is the closest center and we add weight of this point to
                # our center to have more chance to select in k-means++
                if dist_center_point[i, j] == alpha[j]:
                    sum_w_p_i += self.w[j]
            w_p = np.vstack((w_p, sum_w_p_i))

        # Line 14 algorithm 3
        # Create dataset for our centers
        dataset_center = Dataset(self.c)
        kpp_ = KPP(dataset_center)

        return kpp_.fit(self.K, w_p)
