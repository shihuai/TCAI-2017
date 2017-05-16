# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: res_unet2d.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年05月15日 星期一 20时46分18秒
#########################################################################

from keras.utils.visualize_util import plot
#from keras import layers
from keras.models import Model
from keras.layers import Input, Conv2D, merge, UpSampling2D, Activation, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

from metric import dice_coef_loss, dice_coef

class Res_UNet2D(object):

    def __init__(self, color_type = 1, row = 256, col = 256):
        self.color_type = color_type
        self.row = row
        self.col = col
        self.model = self.res_unet2d_model()

    def res_block(self, input1, filter_size = 16, up = False):

        conv1 = input1
        if not up:
            conv1 = Conv2D(filter_size, 3, 3,  subsample = (2, 2), border_mode = 'same')(input1)
            #conv1 = BatchNormalization(axis = 1)(conv1)
            conv1 = Activation('relu')(conv1)

        conv2 = Conv2D(filter_size, 1, 1)(conv1)
        #conv2 = BatchNormalization(axis = 1)(conv2)
        conv2 = Activation('relu')(conv2)
        conv3 = Conv2D(filter_size, 3, 3, border_mode = 'same')(conv2)
        #conv3 = BatchNormalization(axis = 1)(conv3)

        output = merge([conv1, conv3], mode = 'sum', concat_axis = 1)
        output = Activation('relu')(output)

        return output

    def upsampling_block(self, input1, input2, output_filter_size = 16):

        input1 = UpSampling2D(size = (2, 2))(input1)
        up1 = merge([input1, input2], mode = 'concat', concat_axis = 1)
        conv1 = Conv2D(output_filter_size, 1, 1)(up1)
        #conv1 = BatchNormalization(axis = 1)(conv1)
        conv1 = Activation('relu')(conv1)

        return conv1

    def res_unet2d_model(self):
        filter_size = 16

        inputs = Input((self.color_type, self.row, self.col))

        conv1 = Conv2D(filter_size, 3, 3, border_mode='same')(inputs)
        #conv1 = BatchNormalization(axis = 1)(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv2D(filter_size, 3, 3, border_mode = 'same')(conv1)
        #conv2 = BatchNormalization(axis = 1)(conv2)
        conv2 = Activation('relu')(conv2)

        filter_size *= 2
        res_block1 = self.res_block(conv2, filter_size)
        filter_size *= 2
        res_block2 = self.res_block(res_block1, filter_size)
        filter_size *= 2
        res_block3 = self.res_block(res_block2, filter_size)
        filter_size *= 2
        res_block4 = self.res_block(res_block3, filter_size)

        filter_size /= 2
        up_block1 = self.upsampling_block(res_block4, res_block3, filter_size)
        up_block1 = self.res_block(up_block1, filter_size, True)
        filter_size /= 2
        up_block2 = self.upsampling_block(up_block1, res_block2, filter_size)
        up_block2 = self.res_block(up_block2, filter_size, True)
        filter_size /= 2
        up_block3 = self.upsampling_block(up_block2, res_block1, filter_size)
        up_block3 = self.res_block(up_block3, filter_size, True)
        filter_size /= 2
        up_block4 = self.upsampling_block(up_block3, conv2, filter_size)
        up_block4 = self.res_block(up_block4, filter_size, True)

        conv3 = Conv2D(1, 1, 1, activation = 'sigmoid')(up_block4)

        model = Model(input = [inputs], output = [conv3])

        model.compile(optimizer = Adam(lr = 1e-5), loss = dice_coef_loss, metrics = [dice_coef])

        return model

    def train_unet2d(self, trainX, trainY, valX, valY, batch_size = 8, epoch = 10):
        print "trainX shape: ", trainX.shape
        print "trainY shape: ", trainY.shape
        print "valX shape: ", valX.shape
        print "valY shape: ", valY.shape

        for i in range(1):
            self.model.fit(trainX, trainY, batch_size = batch_size, nb_epoch = epoch,
                           verbose = 1, shuffle = True, validation_data = (valX, valY))
        self.model.save_weights('./output/models/res_unet2d_model_' + str(self.row) + 'x' + str(self.col))

    #导入网络模型参数
    def load_mode(self):
        self.model.load_weights('./output/models/res_unet2d_model_' + str(self.row) + 'x' + str(self.col))

    #对测试数据进行预测
    def predict_unet2d(self, testX):
        mask_test = self.model.predict(testX, batch_size = 8, verbose = 1)

        return mask_test

    def show_model(self):
        plot(self.model, to_file = 'res_unet2d.png', show_shapes = True)

def getResUNet2DModel(row, col, color_type):
    return Res_UNet2D(color_type, row, col)

if __name__ == '__main__':
    res_model = Res_UNet2D(1, 256, 256)
    res_model.show_model()
