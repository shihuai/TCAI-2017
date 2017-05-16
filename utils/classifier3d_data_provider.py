# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: classifier3d_data_provider.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年04月26日 星期三 09时57分41秒
#########################################################################

import numpy as np
import cv2
import csv
from glob import glob
import pandas as pd
import os
import random
from keras.utils import np_utils
from utils import normalize, set_window_width

class Classifier3dDataProvider:

    def __init__(self, workspace, data_type):
        self.workspace = workspace
        self.data_type = data_type

    def get_filename(self, file_list, case):
        for f in file_list:
            if case in f:
                return (f)

    def load_data(self, use_data_augment = False):
        dataX = []
        dataY = []
        file_list = glob(os.path.join('./data/nodule_cubes/' + self.data_type + '/npy', '*.npy'))
        for path in file_list:
            name = path.split('/')[-1]
            label = name.split('_')[0]

            dataX.append(np.load(path))
            dataY.append(int(label))

        dataX = np.array(dataX)
        dataY = np.array(dataY)
        dataX = normalize(dataX)
        dataX = dataX.reshape(dataX.shape[0], 1, dataX.shape[1], dataX.shape[2], dataX.shape[3])
        dataY = np_utils.to_categorical(dataY, 2)

        return dataX, dataY

def getClassifier3dDataProvider(workspace, data_type):
    return Classifier3dDataProvider(workspace, data_type)
