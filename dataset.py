import pandas as pd


class Dataset():
    def __init__(self, file: str = None):
        '''
        Just create all the needed variables
        '''
        self.sample = None
        self.number_of_feature = None
        self.number_of_sample = None
        self.number_of_cluster = None
        self.cluster = None
        if(file is not None):
            self.read_dataset(file)

    def read_dataset(self, file: str):
        self.sample = pd.read_csv(file).to_numpy()
        self.number_of_sample = self.sample.shape[0]
        self.number_of_feature = self.sample.shape[1]
