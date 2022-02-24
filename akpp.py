import numpy as np
from binary_heap import Binary_heap
from dataset import Dataset
import heapq

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
        if(self.w is None):
            self.w = np.ones((self.n, 1))
        else:
            self.w = sample_weight.reshape((-1, 1))
        # 1
        landa = np.random.exponential(scale=1.0, size=(self.n, 1))
        # 2
        Q = Binary_heap(landa/self.w)
        # 3
        dirty = np.zeros((self.n, 1), dtype=bool)
        # 4
        self.m = np.vstack((self.m, self.X[Q.pop()]))
        # 5
        aplha = np.full((self.n, 1), np.inf)
        phi = np.zeros((self.n, 1))
        gamma = np.full((self.n, 1), np.inf)
        # 6
        # for k in range(1, self.K):
        #     # 7
        #     for j in range(1, k):
        #         # 8

        print('pause')
        while Q:
            print(Q[0])
            print(heapq.heappop(Q))


dataset = Dataset('dataset/test.csv')
akpp = AKPP(dataset)
akpp.fit(10)
print('pause')
