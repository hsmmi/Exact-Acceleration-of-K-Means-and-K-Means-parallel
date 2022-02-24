import numpy as np
from dataset import Dataset


class AKPP:
    def __init__(self, dataset: Dataset) -> None:
        self.X = dataset.sample
        self.n = dataset.number_of_sample
        self.d = dataset.number_of_sample
        self.w = None
        self.K = None

    def fit(self, number_of_cluster: int, sample_weight: np.ndarray = None):
        assert number_of_cluster < self.n,\
            'number of cluster is greater than number of sample'
        self.K = number_of_cluster
        if(self.w is None):
            self.w = np.ones((self.dataset.number_of_sample, 1))
        else:
            self.w = sample_weight.reshape((-1, 1))
        # 1
        landa = np.random.exponential(scale=1.0, size=(self.n, -1))
        print('pause')


dataset = Dataset('dataset/phishing.csv')
akpp = AKPP(dataset)
akpp.fit(10)
print('pause')
