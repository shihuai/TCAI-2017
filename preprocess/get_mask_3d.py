# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: get_mask_3d.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年04月26日 星期三 15时34分25秒
#########################################################################

import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import scipy.ndimage
import os
import array
import math
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('tqdm 是一个轻量级的进度条小包。。。')
    tqdm = lambda x : x

workspace = '../'


def write_meta_header(filename, meta_dict):
    header = ''
    # do not use tags = meta_dict.keys() because the order of tags matters
    tags = ['ObjectType', 'NDims', 'BinaryData',
            'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize',
            'TransformMatrix', 'Offset', 'CenterOfRotation',
            'AnatomicalOrientation',
            'ElementSpacing',
            'DimSize',
            'ElementType',
            'ElementDataFile',
            'Comment', 'SeriesDescription', 'AcquisitionDate', 'AcquisitionTime', 'StudyDate', 'StudyTime']
    for tag in tags:
        if tag in meta_dict.keys():
            header += '%s = %s\n' % (tag, meta_dict[tag])
    f = open(filename, 'w')
    f.write(header)
    f.close()

def dump_raw_data(filename, data):
    """ Write the data into a raw format file. Big endian is always used. """
    #---将数据写入文件
    # Begin 3D fix
    data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
    # End 3D fix
    rawfile = open(filename, 'wb')
    a = array.array('f')
    for o in data:
        a.fromlist(list(o))
    # if is_little_endian():
    #    a.byteswap()
    a.tofile(rawfile)
    rawfile.close()

def write_mhd_file(mhdfile, data, dsize):
    assert (mhdfile[-4:] == '.mhd')
    meta_dict = {}
    meta_dict['ObjectType'] = 'Image'
    meta_dict['BinaryData'] = 'True'
    meta_dict['BinaryDataByteOrderMSB'] = 'False'
    meta_dict['ElementType'] = 'MET_FLOAT'
    meta_dict['NDims'] = str(len(dsize))
    meta_dict['DimSize'] = ' '.join([str(i) for i in dsize])
    meta_dict['ElementDataFile'] = os.path.split(mhdfile)[1].replace('.mhd', '.raw')
    write_meta_header(mhdfile, meta_dict)
    pwd = os.path.split(mhdfile)[0]
    if pwd:
        data_file = pwd + '/' + meta_dict['ElementDataFile']
    else:
        data_file = meta_dict['ElementDataFile']
    dump_raw_data(data_file, data)

def euclidean_dist(point1, point2):
    A = point1 - point2
    B = A * A

    return math.sqrt(sum(B))

def make_mask(mask, image, v_nodule_center, radius):
    zyx_1 = v_nodule_center - int(radius) + 512
    zyx_2 = v_nodule_center + int(radius) + 1 + 512

    for z in range(int(zyx_1[2]), int(zyx_2[2])):
        for y in range(int(zyx_1[1]), int(zyx_2[1])):
            for x in range(int(zyx_1[0]), int(zyx_2[0])):
#                print z - 512, y - 512, x - 512
#                dist = euclidean_dist(np.array([z - 512, y - 512, x - 512]), v_nodule_center)
#                print z - 512, y - 512, x - 512
#                print v_nodule_center
                if (euclidean_dist(np.array([x - 512, y - 512, z - 512,]), v_nodule_center) <= radius):
                    mask[z - 512, y - 512, x - 512] = 255

#    print 'zyx_1: ', zyx_1
#    print 'zyx_2: ', zyx_2

    zyx_1 -= 512
    zyx_2 -= 512
    print mask[zyx_1[2]:zyx_2[2], zyx_1[1]:zyx_2[1], zyx_1[0]:zyx_2[0]]

    return mask

def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing

def main():
    name = 'LKDS-00415'
    mhd_path = './data/sample_patients/val/' + name + '.mhd'
    img_path = './output/test_masks/CT_mask/' + name + '.npy'
    coords_path = './centroid.csv'

    itk_img = sitk.ReadImage(mhd_path)
    image = sitk.GetArrayFromImage(itk_img)
    origin = np.array(itk_img.GetOrigin())
    spacing = np.array(itk_img.GetSpacing())
    #image, new_spacing = resample(image, spacing)

    f = open(coords_path, 'r')
    nodules_list = f.readlines()
    f.close()

    mask = np.zeros(image.shape)
    print 'CT scan shape: ', image.shape
    print 'Mask scan shape: ', mask.shape
    print 'have %d nodules.' % len(nodules_list)

    for row in nodules_list:
        print row
        coord = row.strip('\n').split(',')

        w_nodule_center = np.array([float(coord[1]), float(coord[2]), float(coord[3])])
        v_nodule_center = np.rint((w_nodule_center - origin) / spacing)
        radius = float(coord[4]) / 2;

        mask = make_mask(mask, image, v_nodule_center, radius)

    mask = mask.astype(np.uint8)

    print 'Make mask done.'

#    mask[mask == 0] = -1000
    #mask = np.load(img_path)
    print 'mask.shape', mask.shape
    print 'image.shape', image.shape

    #mask, new_spacing = resample(mask, spacing)
    print mask.shape
    mask = mask.reshape(mask.shape[2], mask.shape[1], mask.shape[0])
    write_mhd_file('./' + name + '.mhd', mask, mask.shape)
    np.save('./output/test_masks/' + name + '.npy', mask)

if __name__ == '__main__':
    main()

