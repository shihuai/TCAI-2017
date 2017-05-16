# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: data_provider.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年04月22日 星期六 14时11分41秒
#########################################################################

import numpy as np
import cv2
import csv
from glob import glob
import pandas as pd
import os
import random

class UNet2dDataProvider:
    #各个文件路径的初始化操作
    def __init__(self, workspace, data_type):
        self.workspace = workspace
        self.data_type = data_type
        self.all_patients_path = os.path.join(self.workspace, "data/sample_patients", self.data_type)
        self.tmp_workspace = os.path.join(self.workspace, "data/slices_masks", self.data_type)
        self.ls_all_patients = glob(os.path.join(self.all_patients_path, "*.mhd"))
        self.df_annotations = pd.read_csv(os.path.join(self.workspace, "data/csv_files",
                                                       data_type + "/annotations.csv"))
        self.df_annotations["file"] = self.df_annotations["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
        self.df_annotation = self.df_annotations.dropna()

    def get_filename(self, file_list, case):
        for f in file_list:
            if case in f:
                return (f)

    #导入数据
    def load_data(self, row, col, use_data_augment = False):
        X = []
        Y = []
        names = []

        f = open(os.path.join(self.tmp_workspace, 'file_list.csv'), 'r')
        file_list = f.readlines()
        for name in file_list:
            name = name.strip('\n')
            slice_name = name + '.jpg'
            mask_name = name + '_o.jpg'
            slice = cv2.imread(os.path.join(self.tmp_workspace + '/jpg', slice_name), 0)
            mask = cv2.imread(os.path.join(self.tmp_workspace + '/jpg', mask_name), 0)
            mask[mask > 0] = 1

            #判断输入数据的大小是否为256*256
            if slice.shape[0] != row or slice.shape[0] != col:

                slice = cv2.resize(slice, (row, col), interpolation = cv2.INTER_CUBIC)
                mask = cv2.resize(mask, (row, col), interpolation = cv2.INTER_CUBIC)

            if use_data_augment == True:
                X, Y = self.data_augment(X, Y, slice, mask)
            X.append(slice)
            Y.append(mask)
            names.append(name)

        X = np.array(X, dtype = np.float32)
        Y = np.array(Y, dtype = np.uint8)
        print X.shape
        print Y.shape

        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        Y = Y.reshape(Y.shape[0], 1, Y.shape[1], Y.shape[2])
        #X = X / 255.0
        #print trainX[0]

        return names, X, Y

    #对CT切片进行左旋转
    def left_rotate_img(self, slice, mask):
        rows, cols = slice.shape
        rotate_degree = random.uniform(-30, 0)
        M = cv2.getRotationMatrix2D((rows / 2, cols / 2), rotate_degree, 1)
        rotate_slice = cv2.warpAffine(slice, M, (cols, rows))
        rotate_mask  = cv2.warpAffine(mask, M, (cols, rows))

        return rotate_slice, rotate_mask

    #对CT切片进行右旋转
    def right_rotate_img(self, slice, mask):
        rows, cols = slice.shape
        rotate_degree = random.uniform(0, 30)
        M = cv2.getRotationMatrix2D((rows / 2, cols / 2), rotate_degree, 1)
        rotate_slice = cv2.warpAffine(slice, M, (cols, rows))
        rotate_mask  = cv2.warpAffine(mask, M, (cols, rows))

        return rotate_slice, rotate_mask

    #对CT切片进行平移
    def shift_img(self, slice, mask):
        rows, cols = slice.shape

        shift_x = random.uniform(-20, 20)
        shift_y = random.uniform(-20, 20)

        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shift_slice = cv2.warpAffine(slice, M, (rows, cols))
        shift_mask = cv2.warpAffine(mask, M, (rows, cols))

        return shift_slice, shift_mask

    #对CT切片做数据增强操作
    def data_augment(self, X, Y, slice, mask):
        #对CT切片进行左右翻转操作
        X.append(cv2.flip(slice, 1))
        Y.append(cv2.flip(mask, 1))

        rs1, rm1 = self.left_rotate_img(slice, mask)
        rs2, rm2 = self.right_rotate_img(slice, mask)

        ss1, sm1 = self.shift_img(slice, mask)

        X.append(rs1)
        Y.append(rm1)
        X.append(rs2)
        Y.append(rm2)
        X.append(ss1)
        Y.append(sm1)

        return X, Y


def getUNet2dDataProvider(workspace, data_type):
    return UNet2dDataProvider(workspace, data_type)
