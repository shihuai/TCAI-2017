# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: test.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年05月04日 星期四 10时08分11秒
#########################################################################

import SimpleITK as sitk
import numpy as np
import cv2
import os
import sys
import random
from glob import glob
import scipy.ndimage
#sys.path.append("./models_config")
#from segmentation_unet2d import getUNet2DModel
#
#sys.path.append("./utils")
#from utils import normalize, set_window_width

workspace = '../'
rows = 256
cols = 256

def main():
    test_files = glob(os.path.join(workspace, 'data/sample_patients/test/LKDS-00186.mhd'))

    #unet2d_model = getUNet2DModel(rows, cols, 1)
    #unet2d_model.load_mode()

    for patient in test_files:
        full_img_info = sitk.ReadImage(patient)
        full_scan = sitk.GetArrayFromImage(full_img_info)
        ch, oldrows, oldcols = full_scan.shape
        print full_scan.shape

        temp_scan = []
        for slice in full_scan:
            slice = scipy.ndimage.interpolation.zoom(slice,
                                                     [(1.0 * rows) / slice.shape[0],
                                                      (1.0 * cols) / slice.shape[1]],
                                                    mode = 'nearest' )
            temp_scan.append(slice)
        full_scan = np.array(temp_scan)
        print full_scan.shape

        temp_scan = []
        for slice in full_scan:
            slice = scipy.ndimage.interpolation.zoom(slice,
                                                     [(1.0 * oldrows) / slice.shape[0],
                                                      (1.0 * oldcols) / slice.shape[1]],
                                                    mode = 'nearest' )
            temp_scan.append(slice)
        full_scan = np.array(temp_scan)
        print full_scan.shape

if __name__ == '__main__':
    main()
