#coding:utf-8
import numpy as np
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import six
project_path = os.getcwd() + '/../'
sys.path.append(project_path + "utils")

import sklearn.metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.grid_search import GridSearchCV
import sklearn
from sklearn.externals import joblib
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

import seaborn as sns

import function
import preprocessing
import local_bayes
import visualize
import metrics

sys.path.append(project_path + "models/cnn")
import cnn

import data
with open('../original_data/mnist/mnist.pkl', 'rb') as mnist_pickle:
            mnist = six.moves.cPickle.load(mnist_pickle)
            mnist['data'] = mnist['data'].astype(np.float32)
            mnist['data'] /= 255
            mnist['target'] = mnist['target'].astype(np.int32)
            
            N = 60000
            train_x, test_x = np.split(mnist['data'],   [N])
            train_y, test_y = np.split(mnist['target'], [N])
