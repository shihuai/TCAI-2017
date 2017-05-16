# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: unet2d.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年04月23日 星期日 15时13分24秒
#########################################################################

import numpy as np
import cv2
import csv
from glob import glob
import pandas as pd
import os
import sys
import random
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score
#from metric import dice_coef, dice_coef_loss
#from shapely.geometry import MultiPolygon, Polygon
#import shapely.wkt
#import shapely.affinity
from collections import defaultdict
#sys.path.append("./utils")
#from unet2d_data_provider import getDataProvider
from metric import dice_coef_loss, dice_coef, dice_coef_np

class UNet2D(object):
    #网络参数初始化
    def __init__(self, row = 256, col = 256, color_type = 1):
        self.color_type = 1
        self.row = row
        self.col = col
        self.model = self.unet2d_generator()

    #定义一个unet2d网络模型
    def unet2d_generator(self):
        inputs = Input((self.color_type, self.row, self.col))

        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

        up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

        up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

        up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

        up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

        conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)
        #model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
        model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss, metrics = [dice_coef])
        #model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

        return model

    #训练unet2d网络
    def train_unet2d(self, trainX, trainY, valX, valY, batch_size = 8, epoch = 10):

        print "trainX shape: ", trainX.shape
        print "trainY shape: ", trainY.shape
        print "valX shape: ", valX.shape
        print "valY shape: ", valY.shape

        for i in range(1):
            self.model.fit(trainX, trainY, batch_size = 8, epochs = epoch, verbose = 1,
                            shuffle = True, validation_data = (valX, valY))

        self.model.save_weights('./output/models/unet2d_model_' + str(self.row) + 'x' + str(self.col))

    #导入网络模型参数
    def load_mode(self):
        self.model.load_weights('./output/models/unet2d_model_' + str(self.row) + 'x' + str(self.col))

    #对测试数据进行预测
    def predict_unet2d(self, testX):
        mask_test = self.model.predict(testX, batch_size = 8, verbose = 1)

        return mask_test

def getUNet2DModel(row, col, color_type):
    return UNet2D(row, col, color_type)
