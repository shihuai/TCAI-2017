# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: test_canditate_nodules.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年05月08日 星期一 11时22分52秒
#########################################################################

import numpy as np
import os
import math

def get_nodules(path):
    fi = open(path, 'r')
    files = fi.readlines()
    fi.close()

    nodules = {}
    for file in files[1:]:
        row = file.strip('\n').split(',')

        if row[0] not in nodules.keys():
            nodules[row[0]] = []

        nodules[row[0]].append(np.array([float(row[1]), float(row[2]), float(row[3])]))

    return nodules

def euclidean_dist(p1, p2):
    square = (p1 - p2) ** 2

    return math.sqrt(sum(square))

def main():
    annotation_nodules = get_nodules('./data/csv_files/val/annotations.csv')
    candidate_nodules = get_nodules('./output/test_masks/candidate_nodules_centroids.csv')
    #candidate_nodules = get_nodules('./output/submission/submission_file_1.csv')

    pred_count = 0
    total_candidate_nodules = 0
    for patient in annotation_nodules.keys():
        total_candidate_nodules += len(annotation_nodules[patient])
        coords_A = annotation_nodules[patient]
        coords_C = candidate_nodules[patient]

        for p in coords_A:
            for q in coords_C:
                if euclidean_dist(p, q) <= 10.0:
                    pred_count += 1
                    break

    print ('There are {} nodules have been predict.'.format((1.0 * pred_count) / total_candidate_nodules))

if __name__ == "__main__":
    main()
