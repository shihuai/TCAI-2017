# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: classifier_net3d.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年04月25日 星期二 22时06分14秒
#########################################################################

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils
from keras import backend as K

class Classifier3D(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.classifier_3d()

    def classifier_3d(self):
        model = Sequential()

        model.add(Convolution3D(16, 3, 3, 3, input_shape = self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size = (2, 2, 2)))
        model.add(Convolution3D(32, 3, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size = (2, 2, 2)))
        model.add(Convolution3D(64, 3, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size = (2, 2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2))
        model.add(Activation('softmax'))

        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        return model

    def train_classifier_3d(self, trainX, trainY, valX, valY):

        print "trainX shape: ", trainX.shape
        print "trainY shape: ", trainY.shape
        print "valX shape: ", valX.shape
        print "valY shape: ", valY.shape

        for i in range(1):
            self.model.fit(trainX, trainY, batch_size = 16, nb_epoch = 100, verbose = 1,
                           shuffle = True, validation_data = (valX, valY))

        self.model.save_weights('./output/models/classifier_3d_model')

    def load_mode(self):
        self.model.load_weights('./output/models/classifier_3d_model')

    def predict_classifier_3d(self, testX):
        pred_score = self.model.predict(testX, batch_size = 16, verbose = 1)

        return pred_score

def getClassifier3DModel(x, y, z, color_type):
    return Classifier3D([color_type, z, x, y])

