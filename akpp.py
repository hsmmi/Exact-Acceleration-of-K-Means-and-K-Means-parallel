from collections import deque
from tools import distance
import numpy as np
from binary_heap import Binary_heap
from dataset import Dataset


class AKPP:
    def __init__(self, dataset: Dataset) -> None:
        self.X = dataset.sample
        self.n = dataset.number_of_sample
        self.d = dataset.number_of_feature
        self.w = None
        self.K = None
        self.m = np.empty((0, self.d))

    def fit(self, number_of_cluster: int, sample_weight: np.ndarray = None):
        assert number_of_cluster < self.n,\
            'number of cluster is greater than number of sample'
        self.K = number_of_cluster
        self.m = np.empty((0, self.d))
        if(sample_weight is None):
            self.w = np.ones((self.n, 1))
        else:
            self.w = sample_weight.reshape((-1, 1))
        # line 1 algorithm 2
        landa = np.random.exponential(scale=1, size=(self.n, 1))
        # line 2 algorithm 2
        Q = Binary_heap(landa/self.w)
        # line 3 algorithm 2
        dirty = np.zeros((self.n, 1), dtype=bool)
        # line 4 algorithm 2
        self.m = np.vstack((self.m, self.X[Q.pop()]))
        # line 5 algorithm 2
        alpha = np.full((self.n, 1), np.inf)
        phi = np.zeros((self.n, 1))
        gamma = np.full((self.n, 1), np.inf)
        # line 6 algorithm 2
        for k in range(self.K - 1):
            # line 7 & 8 algorithm 2
            gamma[:k] = distance(self.m[:-1], self.m[-1])
            # line 9 algorithm 2
            for i in range(self.n):
                # line 10 & 11 algorithm 2
                if(gamma[int(phi[i][0])] >= 2 * alpha[i]):
                    continue
                # line 12 - 14 algorithm 2
                dis_mk_xi = distance(self.m[k], self.X[i])
                if(dis_mk_xi < alpha[i]):
                    alpha[i] = dis_mk_xi
                    phi[i][0] = k
                    dirty[i] = True
            # line 15 - 18 algorithm 2
            S = deque()
            while dirty[Q.peek()]:
                i = Q.pop()
                S.append(i)
            # line 19 - 21 algorithm 2
            for i in S:
                Q.push(landa[i]/(self.w[i]*(alpha[i]**2)))
                dirty[i] = False
            # line 22 algorithm 2
            self.m = np.vstack((self.m, self.X[Q.pop()]))
        # line 23 algorithm 2
        return self.m
