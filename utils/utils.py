# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: utils/utils.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年04月27日 星期四 15时56分48秒
#########################################################################

import numpy as np
import scipy.ndimage
import array
import math

#---数据标准化
def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    """数据标准化"""
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

#---设置窗宽
def set_window_width(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    """设置窗宽"""
    image[image > MAX_BOUND] = MAX_BOUND
    image[image < MIN_BOUND] = MIN_BOUND
    return image
#---重采样
def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing
