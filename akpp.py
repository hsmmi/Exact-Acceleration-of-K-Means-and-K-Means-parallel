from tools import distance, execution_time
import numpy as np
from binary_heap import BinaryHeap
from dataset import Dataset


class AKPP:
    def __init__(self, dataset: Dataset) -> None:
        self.X = dataset.sample
        self.n = dataset.number_of_sample
        self.d = dataset.number_of_feature
        self.w = None
        self.K = None
        self.m = np.empty((0, self.d))

    @execution_time
    def fit(self, number_of_cluster: int, sample_weight: np.ndarray = None):
        assert (
            number_of_cluster < self.n
        ), "number of cluster is greater than number of sample"
        self.K = number_of_cluster
        self.m = np.empty((0, self.d))
        if sample_weight is None:
            self.w = np.ones((self.n, 1))
        else:
            self.w = sample_weight.reshape((-1, 1))
        # Line 1 algorithm 2
        # Using landa for randomnes
        landa = np.random.exponential(scale=1, size=(self.n, 1))
        # Line 2 algorithm 2
        # Priority queue using a standard binary heap
        # highest priority (smallest values)
        # Build in O(log n)
        Q = BinaryHeap(landa / self.w)
        # Line 3 algorithm 2
        # The item has a higher priority than it actually should
        dirty = np.zeros((self.n, 1), dtype=bool)
        # Line 4 algorithm 2
        # Select highest piority node and remove from tree with O(log n)
        self.m = np.vstack((self.m, self.X[Q.pop()]))
        # Line 5 algorithm 2
        # Distance to closest mean
        alpha = np.full((self.n, 1), np.inf)
        # Id of closest mean
        # At first all nodes assign to m[0] (the only mean)
        phi = np.zeros((self.n, 1))
        # Line 6 algorithm 2
        for k in range(self.K - 1):
            # Line 7 & 8 algorithm 2
            # Distance of prevous centers to last center (k-1)x1
            gamma = distance(self.m, self.m[-1])
            # Line 9 algorithm 2
            for i in range(self.n):
                # Line 10 & 11 algorithm 2
                # With triangle inequality if x be a point
                # and b and c be centers if d(b, c) >= 2d(x, b)
                # then d(x, c) >= d(x, b) so we continue
                if gamma[int(phi[i][0])] >= 2 * alpha[i]:
                    continue
                # Line 12 - 14 algorithm 2
                dis_mk_xi = distance(self.m[k], self.X[i])
                # Update if we current center is closer than before
                if dis_mk_xi < alpha[i]:
                    alpha[i] = dis_mk_xi
                    phi[i][0] = k
                    # Now as alpha decrease it's piority decrease(higher value)
                    # so it become dirty and we should update it later
                    dirty[i] = True
            # Line 15 - 18 algorithm 2
            S = []
            # We only need reprioritize untill we find clean node
            # because all the nodes after that not have a chance to
            # be the highest piority because the piority only decrease
            # so we add dirty node with highes piority to queue to reprioritize
            # untill we reach clean node
            while Q.heap and dirty[Q.peek()]:
                i = Q.pop()
                S.append(i)
            # Line 19 - 21 algorithm 2
            # Now we reprioritize all the nodes in queue and make them clean
            # If all the nodes was dirty now we use another O(log n) to
            # build a binary heap tree
            piority = np.divide(landa[S], (self.w[S] * (alpha[S] ** 2)))
            piority[np.isnan(piority)] = np.inf
            for indx, i in enumerate(S):
                Q.push(piority[indx], i)
                dirty[i] = False
            # Line 22 algorithm 2
            # Select next mean with highest piority
            self.m = np.vstack((self.m, self.X[Q.pop()]))
        # Line 23 algorithm 2
        return self.m
