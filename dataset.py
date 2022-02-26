import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, file: str or np.ndarray = None):
        """ Just create all the needed variables

        To have and store all thing related to dataset, I create this class

        If you gave it str as file, it'll read dataset from file path
        or if file be ndarray, it'll use the array as samples
        or file can be None so in this case you can use read_dataset function

        Args:
            file (str|ndarray)
        """
        self.sample = None
        self.number_of_feature = None
        self.number_of_sample = None
        if file is not None:
            self.read_dataset(file)

    def read_dataset(self, file: str or np.ndarray):
        """ Store the data in class

        If you gave it str as file, it'll read dataset from file path
        or if file be ndarray, it'll use the array as samples

        Args:
            file (str|ndarray)
        """
        if type(file) == str:
            self.sample = pd.read_csv(file).to_numpy()
            self.number_of_sample = self.sample.shape[0]
            self.number_of_feature = self.sample.shape[1]
        else:
            self.sample = file
            self.number_of_sample = file.shape[0]
            self.number_of_feature = file.shape[1]
