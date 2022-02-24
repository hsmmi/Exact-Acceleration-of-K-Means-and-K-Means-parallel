import os
from matplotlib import pyplot as plt
import numpy as np
import cv2


class Dataset():
    def __init__(self):
        '''
        Just create all the needed variables
        '''
        self.image_size = None
        self.sample = None
        self.mean_sample = None
        self.label = None
        self.diffrent_label = None
        self.count_diffrent_label = None
        self.number_of_diffrent_label = None
        self.label_name = None
        self.number_of_feature = None
        self.number_of_sample = None

    def read_dataset(self, folder: str, img_size: int,
                     visualize: bool = False):

        images = []
        feelings = []

        directory = os.path.abspath('')
        folder_path = os.path.join(directory, folder)
        for img_name in os.listdir(folder_path):
            feelings.append(img_name[3:5])
            img_path = os.path.join(folder_path, img_name)
            img_mat = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_mat is not None:
                resized_img_mat = cv2.resize(img_mat, (img_size, img_size))
                images.append(resized_img_mat)

        images = np.array(images)
        feelings = np.array(feelings).reshape((-1, 1))

        if visualize is True:
            for rnd in self.visualize_sample:
                plt.imshow(images[rnd],  cmap="gray")
                plt.title((f'feeling: {feelings[rnd]}'))
                plt.show()

        vector_images = images.reshape(images.shape[0], -1)

        self.image_size = img_size
        self.sample = vector_images
        self.label = feelings
        self.diffrent_label, self.count_diffrent_label = np.unique(
            self.label, return_counts=True)
        self.number_of_diffrent_label = self.diffrent_label.shape[0]
        self.label_name = np.array(['feeling'])
        self.number_of_feature = vector_images.shape[1]
        self.number_of_sample = vector_images.shape[0]
