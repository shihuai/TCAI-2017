# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: train.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年04月19日 星期三 19时59分17秒
#########################################################################

import SimpleITK as sitk
import numpy as np
import cv2
import csv
from glob import glob
import pandas as pd
import os
import sys
import random
import scipy.ndimage
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score
#from metric import dice_coef, dice_coef_loss
#from shapely.geometry import MultiPolygon, Polygon
#import shapely.wkt
#import shapely.affinity
from collections import defaultdict
sys.path.append("utils")
from unet2d_data_provider import getUNet2dDataProvider

workspace = './'
smooth = 1e-12

#损失函数
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

#定义模型结构
def unet_model():

    inputs = Input((1, 256, 256))

    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    #conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    #conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    #conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    #conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    #conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Conv2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    #model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss, metrics = [dice_coef])
    #model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model


#训练网络模型
def train_net(trainX, trainY, valX, valY, use_exiting):
    model = unet_model()
    #model_checkpoint = ModelCheckpoint('./output/models/unet2d_model_256x256', monitor = 'loss', save_best_only = True)

    #if use_exiting:
    #    model.load_weights('./output/models/unet2d_model_256x256')

    print 'trainX shape: ', trainX.shape
    print 'trainY shape: ', trainY.shape
    print 'valX shape: ', valX.shape
    print 'valY shape: ', valY.shape

    for i in range(1):
        #model.fit(trainX, trainY, batch_size = 8, nb_epoch = 50, verbose = 1,
        #          shuffle = True, callbacks = [model_checkpoint],
        #          validation_data = (valX, valY))
        model.fit(trainX, trainY, batch_size = 8, nb_epoch = 100, verbose = 1,
                  shuffle = True, validation_data = (valX, valY))

        model.save_weights('./output/models/unet_test_model')

    return model

#用训练好的模型对新的CT图像进行预测
def predict_test(names, testX):
    model = unet_model()
    model.load_weights('./output/models/unet_test_model')

    mask_test = model.predict(testX, batch_size = 8, verbose = 1)

    np.save('./mask.npy', mask_test[57])
    #mask_test = 255.0 * mask_test
    #mask_test = mask_test.astype(np.uint8)
    #mask = mask_test[57]
    #print mask.shape
    #fwrite = open('mask.txt', 'w')
    #for x in range(mask.shape[1]):
    #    for y in range(mask.shape[2]):
    #        fwrite.write(str(mask[0, x, y]) + ' ')
    #    fwrite.write('\n')
    #fwrite.close()

    #print mask_test.shape
    #保存预测结果
    for idx, name in enumerate(names):
        mask = mask_test[idx]
        mask[mask >= 0.5] = 255
        mask[mask < 0.5] = 0

        mask = mask.reshape(mask.shape[1], mask.shape[2], mask.shape[0])
        mask = mask.astype(np.uint8)
        img = testX[idx].reshape(testX[idx].shape[1], testX[idx].shape[2], testX[idx].shape[0])
        cv2.imwrite("./output/test_masks/jpg/img/" + name + ".jpg", img)
        cv2.imwrite("./output/test_masks/jpg/mask/" + name + "_o.jpg", mask)

    mask_test[mask_test >= 0.5] = 255.
    mask_test[mask_test < 0.5] = 0.
    mask_test = mask_test.astype(np.uint8)
    mask_test = mask_test[:, 0, :, :]

    np.save('./output/test_masks/LKDS-00012.npy', mask_test)

    return mask_test

def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    """数据标准化"""
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def set_window_width(image, MIN_BOUND = -1000.0, MAX_BOUND = 400.0):
    image[image < MIN_BOUND] = MIN_BOUND
    image[image > MAX_BOUND] = MAX_BOUND

    return image

def load_ct_scan():
    path = 'data/sample_patients/test/LKDS-00012.mhd'
    full_image_info = sitk.ReadImage(path)
    full_scan = sitk.GetArrayFromImage(full_image_info)

    names = []
    slices = []
    for idx, slice in enumerate(full_scan):
        names.append(str(idx))
        slice = scipy.ndimage.interpolation.zoom(slice, [0.5, 0.5], mode = 'nearest')
        slice = normalize(slice)
        slice = 255.0 * slice
        slices.append(slice.astype(np.uint8))

    slices = np.array(slices, dtype = np.float32)
    slices = slices.reshape(slices.shape[0], 1, slices.shape[1], slices.shape[2])

    return names, slices

if __name__ == '__main__':

    #mean = np.load('./output/norm/mean.npy')
    #std = np.load('./output/norm/std.npy')

    train_provider = getUNet2dDataProvider(workspace, 'train')
    trainName, trainX, trainY = train_provider.load_data(256, 256, True)

    #mean = np.mean(trainX)
    #std = np.std(trainX)
    #trainX -= mean
    #trainX /= std

    val_provider = getUNet2dDataProvider(workspace, 'val')
    valName, valX, valY = val_provider.load_data(256, 256)

    #valName, valX = load_ct_scan()
    #valX -= mean
    #valX /= std

    model = train_net(trainX, trainY, valX, valY, False)
    mask_test_pred = predict_test(valName, valX)

    #np.save('./output/norm/mean.npy', mean)
    #np.save('./output/norm/std.npy', std)

    #mean = 0.0
    #for i in range(valY.shape[0]):
    #    mean += dice_coef_np(mask_test_pred[i, 0], valY[i, 0])

    #mean /= valX.shape[0]
    #print ('Mean Dice Coeff: ', mean)
