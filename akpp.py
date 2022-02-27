from collections import deque
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
    def fit(self, K: int, w: np.ndarray = None):
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

        # line 1 algorithm 2
        landa = np.random.exponential(scale=1, size=(self.n, 1))
        # line 2 algorithm 2
        Q = BinaryHeap(landa / self.w)
        # line 3 algorithm 2
        dirty = np.zeros((self.n, 1), dtype=bool)
        # line 4 algorithm 2
        self.m = np.vstack((self.m, self.X[Q.pop()]))
        # line 5 algorithm 2
        alpha = np.full((self.n, 1), np.inf)
        phi = np.zeros((self.n, 1))
        # line 6 algorithm 2
        for k in range(self.K - 1):
            # line 7 & 8 algorithm 2
            gamma = distance(self.m[-1], self.m[:-1])
            # line 9 algorithm 2
            for i in range(self.n):
                # line 10 & 11 algorithm 2
                if gamma[int(phi[i][0])] >= 2 * alpha[i]:
                    continue
                # line 12 - 14 algorithm 2
                dis_mk_xi = distance(self.m[k], self.X[i])
                if dis_mk_xi < alpha[i]:
                    alpha[i] = dis_mk_xi
                    phi[i][0] = k
                    dirty[i] = True
            # line 15 - 18 algorithm 2
            S = deque()
            while Q.heap and dirty[Q.peek()]:
                i = Q.pop()
                S.append(i)
            # line 19 - 21 algorithm 2
            for i in S:
                Q.push(landa[i] / (self.w[i] * (alpha[i] ** 2)), i)
                dirty[i] = False
            # line 22 algorithm 2
            self.m = np.vstack((self.m, self.X[Q.pop()]))
        # line 23 algorithm 2
        return self.m
