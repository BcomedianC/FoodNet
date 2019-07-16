import os
from os import listdir
from os.path import isfile, join
import shutil
import stat
import collections
from collections import defaultdict

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from scipy.misc import imresize

import h5py
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model

from datasets import Datasets

class Preprocessing(object):
    """
    This class manages the construction of the dataset. Any data cleaning, pipelining,
    and reading from disk is performed here.
    """
    def __init__(self):
        pass

    
    """
    Loads the data into disk using the Datasets utility class.
    """
    def load_data(self):
        datasets = Datasets('http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz')
        datasets.download("food101.tgz")


Preprocessing().load_data()