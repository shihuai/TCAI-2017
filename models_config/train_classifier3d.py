# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: train_classifier3d.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年04月26日 星期三 10时27分23秒
#########################################################################

import numpy as np
import cv2
import os
import sys
import random

sys.path.append("./utils")
from classifier3d_data_provider import getClassifier3dDataProvider
from utils import resample, normalize, set_window_width

sys.path.append("./models_config")
from classifier_net3d import getClassifier3DModel

workspace = './'

def main():
    train_data_provider = getClassifier3dDataProvider(workspace, 'train')
    trainX, trainY = train_data_provider.load_data()

    val_data_provider = getClassifier3dDataProvider(workspace, 'val')
    valX, valY = val_data_provider.load_data()

    net3d_model = getClassifier3DModel(45, 45, 45, 1)
    net3d_model.train_classifier_3d(trainX, trainY, valX, valY)

    pred_score = net3d_model.predict_classifier_3d(valX)


if __name__ == '__main__':
    main()
