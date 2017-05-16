# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: train.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年04月23日 星期日 15时36分29秒
#########################################################################

import numpy as np
import cv2
import os
import sys
import random
sys.path.append("./utils")
from unet2d_data_provider import getUNet2dDataProvider
sys.path.append("./models_config")
from segmentation_unet2d import getUNet2DModel
from res_unet2d import getResUNet2DModel
from metric import dice_coef_np

workspace = './'

def save_pred_mask(names, mask_test):
    for idx, name in enumerate(names):
        mask = mask_test[idx]
        mask[mask > 0.5] = 255
        mask[mask < 0.5] = 0
        mask = mask.reshape(mask.shape[1], mask.shape[2], mask.shape[0])
        cv2.imwrite("./output/test_masks/jpg/" + name + ".jpg", mask)
        #np.save("./output/test_masks/npy/" + name + ".npy", mask_test[idx])

def main():
    train_data_provider = getUNet2dDataProvider(workspace, 'train')
    train_names, trainX, trainY = train_data_provider.load_data(256, 256, True)

    val_data_provider = getUNet2dDataProvider(workspace, 'val')
    val_names, valX, valY = val_data_provider.load_data(256, 256)

    #unet2d_model = getUNet2DModel(256, 256, 1)
    unet2d_model = getResUNet2DModel(256, 256, 1)
    unet2d_model.train_unet2d(trainX, trainY, valX, valY, 8, 100)

    mask_test = unet2d_model.predict_unet2d(valX)

    save_pred_mask(val_names, mask_test)
    mean = 0.0
    for i in range(valX.shape[0]):
        mean += dice_coef_np(mask_test[i, 0], valY[i, 0])

    mean /= valX.shape[0]

    print ("Mean Dice Coeff: ", mean)

if __name__ == '__main__':
    main()

