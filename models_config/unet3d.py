# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: unet3d.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年05月11日 星期四 20时39分58秒
#########################################################################

from keras.models import Model
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D, Cropping3D
from keras.optimizers import Adam
from metric import dice_coef_loss, dice_coef

class UNet3D(object):
    def __init__(self, depth = 64, height = 64, width = 64, color_type = 1):
        self.color_type = color_type
        self.depth = depth
        self.height = height
        self.width = width
        self.model = self.unet3d_generator()

    def unet3d_generator(self):
        inputs = Input((self.color_type, self.depth, self.height, self.width))

        conv1 = Convolution3D(128, 3, 3, 3, activation = 'relu')(inputs)
        conv2 = Convolution3D(128, 3, 3, 3, activation = 'relu')(conv1)
        conv3 = Convolution3D(128, 3, 3, 3, activation = 'relu')(conv2)
        pool1 = MaxPooling3D(pool_size = (2, 2, 2))(conv3)

        conv4 = Convolution3D(128, 3, 3, 3, activation = 'relu')(pool1)
        conv5 = Convolution3D(128, 3, 3, 3, activation = 'relu')(conv4)
        conv6 = Convolution3D(128, 3, 3, 3, activation = 'relu')(conv5)
        conv7 = Convolution3D(128, 3, 3, 3, activation = 'relu')(conv6)

        up1 = merge([UpSampling3D(size = (2, 2, 2))(conv7),
                     Cropping3D(cropping = ((8, 8), (8, 8), (8, 8)))(conv3)], mode = 'concat', concat_axis = 1)
        conv8 = Convolution3D(256, 3, 3, 3, activation = 'relu')(up1)
        conv9 = Convolution3D(128, 3, 3, 3, activation = 'relu')(conv8)
        conv10 = Convolution3D(128, 3, 3, 3, activation = 'relu')(conv9)
        conv11 = Convolution3D(128, 3, 3, 3, activation = 'relu')(conv10)
        conv12 = Convolution3D(128, 3, 3, 3, activation = 'relu')(conv11)

        model = Model(input = inputs, output = conv12)

        model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss, metric = [dice_coef])

        return model
