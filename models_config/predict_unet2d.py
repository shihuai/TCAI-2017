# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: predict.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年04月25日 星期二 14时18分55秒
#########################################################################

import SimpleITK as sitk
import numpy as np
import cv2
import os
import sys
import random
from glob import glob
import scipy.ndimage
sys.path.append("./models_config")
from segmentation_unet2d import getUNet2DModel

sys.path.append("./utils")
from utils import normalize, set_window_width

workspace = './'
rows = 256
cols = 256

def main():
    test_files = glob(os.path.join(workspace, 'data/sample_patients/val/*.mhd'))
    #mean = np.load('./output/norm/mean.npy')
    #std = np.load('./output/norm/std.npy')

    unet2d_model = getUNet2DModel(rows, cols, 1)
    unet2d_model.load_mode()

    for idx, patient in enumerate(test_files):
        full_img_info = sitk.ReadImage(patient)
        full_scan = sitk.GetArrayFromImage(full_img_info)
        ch, old_rows, old_cols = full_scan.shape

        if old_rows != rows or old_cols != cols:
            temp_scan = []
            for slice in full_scan:
                slice = scipy.ndimage.interpolation.zoom(slice,
                                                         [(1.0 * rows) / slice.shape[0],
                                                          (1.0 * cols) / slice.shape[1]],
                                                        mode = 'nearest' )
                temp_scan.append(slice)

            full_scan = np.array(temp_scan)

        full_scan = full_scan.reshape(full_scan.shape[0], 1,
                                      full_scan.shape[1], full_scan.shape[2])
        full_scan = normalize(full_scan)
        full_scan = full_scan * 255.0
        full_scan = full_scan.astype(np.float32)
        #full_scan -= mean
        #full_scan /= std

        mask_pred = unet2d_model.predict_unet2d(full_scan)
        mask_pred = mask_pred[:, 0, :, :]
        mask_pred[mask_pred >= 0.5] = 255.
        mask_pred[mask_pred < 0.5] = 0.
        mask_pred = mask_pred.astype(np.uint8)

        if old_rows != mask_pred.shape[1] or old_cols != mask_pred.shape[2]:
            temp_scan = []
            for slice in mask_pred:
                slice = scipy.ndimage.interpolation.zoom(slice,
                                                         [(1.0 * old_rows) / slice.shape[0],
                                                          (1.0 * old_cols) / slice.shape[1]],
                                                        mode = 'nearest' )
                temp_scan.append(slice)

            mask_pred = np.array(temp_scan)

        split_path = patient.split('/')
        name = split_path[-1].split('.')[0]
        np.save(os.path.join(workspace, 'output/test_masks/CT_mask/' + name + '.npy'), mask_pred)
        print ('{}/{} : {} have been preprocessed.'.format(idx, len(test_files), patient))

if __name__ == '__main__':
    main()

