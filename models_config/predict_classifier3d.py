# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: predict_classifier3d.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年04月27日 星期四 15时38分33秒
#########################################################################

import SimpleITK as sitk
import numpy as np
import os
import sys
import math
from classifier_net3d import getClassifier3DModel

sys.path.append('./utils/')
from utils import resample, normalize, set_window_width

def load_test_data():
    path = './output/test_masks/candidate_nodules_centroids.csv'
    file = open(path, 'r')
    candidate_nodules_list = file.readlines()
    file.close()

    nodules_coords = {}
    nodules_data = {}
    for line in candidate_nodules_list:
        row = line.strip('\n').split(',')

        if row[0] not in nodules_coords.keys():
            nodules_coords[row[0]] = []

        nodules_coords[row[0]].append(np.array([float(row[3]), float(row[2]), float(row[1])]))

    new_nodules_coords = {}
    for idx, patient in enumerate(nodules_coords.keys()):
        print ('{}/{} start preprocess: {}'.format(idx, len(nodules_coords), patient))

        ct_scan_path = './data/sample_patients/test/' + patient + '.mhd'
        itk_img = sitk.ReadImage(ct_scan_path)
        image = sitk.GetArrayFromImage(itk_img)
        origin = np.array(itk_img.GetOrigin())[::-1]
        old_spacing = np.array(itk_img.GetSpacing())[::-1]
        image, new_spacing = resample(image, old_spacing)

        nodules = []
        for idx, nodule_coord in enumerate(nodules_coords[patient]):
            v_center = np.rint((nodule_coord - origin) / new_spacing)
            v_center = np.array(v_center, dtype = int)

            window_size = 17
            zyx_1 = v_center - window_size
            zyx_2 = v_center + window_size + 1
            nodule_box = np.zeros([45, 45, 45], np.int16)
            img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
            #img_crop = set_window_width(img_crop)
            img_crop = normalize(img_crop)
            zeros_fill = int(math.floor((45 - (2 * window_size + 1)) / 2))

            try:
                nodule_box[zeros_fill:45 - zeros_fill, zeros_fill:45 - zeros_fill, zeros_fill:45 - zeros_fill] = img_crop
            except:
                continue

            if patient not in new_nodules_coords.keys():
                new_nodules_coords[patient] = []

            new_nodules_coords[patient].append(nodule_coord)
            nodules.append(nodule_box)

        nodules_data[patient] = nodules

    return new_nodules_coords, nodules_data

def main():
    nodules_coords, nodules_data = load_test_data()
    classifier_model = getClassifier3DModel(45, 45, 45, 1)
    classifier_model.load_mode()

    nodules_scores = {}
    for patient in nodules_coords.keys():
        testX = nodules_data[patient]
        testX = np.array(testX)
        testX = testX.reshape(testX.shape[0], 1, testX.shape[1], testX.shape[2], testX.shape[3])
        pred_score = classifier_model.predict_classifier_3d(testX)

        nodules_scores[patient] = pred_score

    file = open('./output/submission/submission_file.csv', 'w')
    file.write('seriesuid,coordX,coordY,coordZ,probability\n')
    for patient in nodules_coords.keys():
        for idx, label in enumerate(nodules_scores[patient].argmax(axis = 1)):
            coord = nodules_coords[patient][idx]
            score = nodules_scores[patient][idx, 1]

            if label == 1:
                file.write(patient + ',' + str(coord[2]) + ',' + str(coord[1]) +
                           ',' + str(coord[0]) + ',' + str(score))
                file.write('\n')

    file.close()

if __name__ == '__main__':
    main()
