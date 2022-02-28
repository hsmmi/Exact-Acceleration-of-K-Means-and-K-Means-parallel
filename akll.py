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
        We'll find seed with Accelerated K-Means|| method

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
        ), "size weights should be nx1(number of sample)"

        # Line 1 algorithm 5
        # Probability to select next center
        beta = self.w / np.sum(self.w)
        # Line 2 algorithm 5
        # Select first center with probabiliry beta
        self.c = np.vstack((self.c, new_seed(self.X, 1, beta)))
        # Line 3 algorithm 5
        # alpha: Distance to closest mean
        alpha = np.full((self.n, 1), np.inf)
        # In each iteration new centers are in c(k_pre, k]
        # k always point to last center
        k_pre, k = -1, 0
        # Line 4 algorithm 5
        for r in range(R):
            # Do we create center in previous round?
            if k - (k_pre + 1) + 1 > 0:
                # Line 5 algorithm 5
                # Biuld VPTree from our centers created in previous round
                # with their index
                C = NNS(
                    self.c[k_pre + 1: k + 1],
                    np.arange(start=k_pre + 1, stop=k + 1),
                )
                # Line 6 - 9 algorithm 5
                # For each sample
                for i in range(self.n):
                    # Find closest center in centers created in previous round
                    # in range alpha[i] to update alpha[i]
                    dis, j = C.nearest_in_range(self.X[i], alpha[i])
                    # Was there a center closer than alpha[i]
                    # If yes j has it's index and distance
                    # in dis otherwise j is -1
                    if j >= 0:
                        alpha[i] = dis
            # Line 10 algorithm 5
            # Now we want to create new centers
            # So now our next last previous center
            # is k which is current last center
            k_pre = k
            # Normalize the probability of selection
            Z = np.sum(self.w * (alpha**2))
            # Line 11 algorithm 5
            for i in range(self.n):
                # Expect to select L new centers
                p = min(1, self.L * self.w[i] * (alpha[i] ** 2) / Z)
                # Line 12 algorithm 5
                # Is point selected as center?
                if p > np.random.rand(1)[0]:
                    # Line 14 algorithm 5
                    # X[i] become new center
                    k = k + 1
                    self.c = np.vstack((self.c, self.X[i]))
                    # Now distance this point to closest center is zere(itself)
                    # As alpha[i] become zero the probability of selecting this
                    # point in next round gets zero so each point select once
                    alpha[i] = 0

        # Line 14 algorithm 5
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

        # Line 15 algorithm 5
        # Create dataset for our centers
        dataset_center = Dataset(self.c)
        kpp = KPP(dataset_center)
        # Select K centers from our many centers (around 2xKxR)
        return kpp.fit(self.K, w_p)
