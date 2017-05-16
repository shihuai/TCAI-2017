# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: metric.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年04月23日 星期日 15时07分56秒
#########################################################################

import numpy as np
from keras import backend as K

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
