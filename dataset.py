import numpy as np
import pandas as pd


class Dataset():
    def __init__(self, file: str or np.ndarray = None):
        '''
        Just create all the needed variables
        '''
        self.sample = None
        self.number_of_feature = None
        self.number_of_sample = None
        if(file is not None):
            self.read_dataset(file)

    def read_dataset(self, file: str or np.ndarray):
        if type(file) == str:
            self.sample = pd.read_csv(file).to_numpy()
            self.number_of_sample = self.sample.shape[0]
            self.number_of_feature = self.sample.shape[1]
        else:
            self.sample = file
            self.number_of_sample = file.shape[0]
            self.number_of_feature = file.shape[1]
