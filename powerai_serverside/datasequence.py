# Credit for this code goes to: https://techblog.appnexus.com/a-keras-multithreaded-dataframe-generator-for-millions-of-image-files-84d3027f6f43

import os
import math
import random

from keras.utils import Sequence
from keras.preprocessing.image import load_img, img_to_array

import numpy as np

datapath = './im/'

def load_image(im):
    return img_to_array(load_img(im, target_size=(224, 224))) / 255.

class DataSequence(Sequence):
    """
    Keras Sequence object to train a model on larger-than-memory data.
    """
    def __init__(self, df, data_path, batch_size, mode='train'):
        self.df = df
        self.bsz = batch_size
        self.mode = mode

        # Take labels and a list of image locations in memory
        self.labels = self.df['label'].values
        self.im_list = self.df['image_name'].apply(lambda x: os.path.join(data_path, x)).tolist()

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.im_list))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return self.labels[idx * self.bsz: (idx + 1) * self.bsz]

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return np.array([load_image(im) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y
