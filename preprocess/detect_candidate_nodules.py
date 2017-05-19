# -*- coding:UTF-8 -*-
# !/usr/bin/env python

#########################################################################
# File Name: get_candidate_nodules.py
# Author: Banggui
# mail: liubanggui92@163.com
# Created Time: 2017年04月26日 星期三 11时20分01秒
#########################################################################

import SimpleITK as sitk
import numpy as np
from glob import glob
import os
import math


def BFS(masks):
    map3d = masks.copy()
    ch, height, width = masks.shape
    visited = np.ones(masks.shape, dtype = np.uint8)
    visited[masks == 255.] = 0
    steps = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    count = 1

    for z in range(ch):
        if np.any(map3d[z] == 255.):
            yx = np.where(map3d[z] == 255.)
            coord = [z, yx[1][0], yx[0][0]]
            queue = []
            queue.append(coord)
            visited[coord[0], coord[2], coord[1]] = 1.

            while len(queue) > 0:
                coord = queue[0]
                del queue[0]

                for step in steps:
                    z = coord[0] + step[0]
                    y = coord[2] + step[2]
                    x = coord[1] + step[1]
                    if (z >= 0 and z < ch) and (y >= 0 and y < height) \
                            and (x >= 0 and x < width) and (visited[z, y, x] == 0):
                        visited[z, y, x] = 1
                        map3d[z, y, x] = count
                        queue.append([z, y, x])

            count += 1

    return map3d, count

#获取阈值大于255.的位置的点作为结节内部的点
def get_coords(mask, threshold):

    map3d, nodule_num = BFS(mask)
    ch, height, width = map3d.shape

    coords = []
    for num in range(1, nodule_num):
        for z in range(1, ch - 1):
            if np.any(map3d[z] == num) and np.any(map3d[z + 1] != num):
                if np.any(map3d[z - 1] != num):
                    temp = map3d[z]
                    temp[temp == num] = 0.
                    map3d[z] = temp
                    break

    for num in range(1, nodule_num):
        index_list = np.where(map3d == num)
        if len(index_list[0]) > 0:
            temp_coord = []
            first_axis = index_list[0]
            for idx, i in enumerate(first_axis):
                temp_coord.append(np.array([index_list[2][idx], index_list[1][idx], index_list[0][idx]]))
            coords.append(temp_coord)

    coords = np.array(coords)
    #print 'candidate coords shape: ', coords.shape

    return coords

def get_centroids(coords):
    cluster_centroids = []
    print coords.shape

    for nodule in coords:
        num = len(nodule)
        x = 0.0
        y = 0.0
        z = 0.0

        for coord in nodule:
            x += coord[0]
            y += coord[1]
            z += coord[2]

        x = (1.0 * x) / num
        y = (1.0 * y) / num
        z = (1.0 * z) / num

        cluster_centroids.append([x, y, z])

    cluster_centroids = np.array(cluster_centroids)
    print 'have {} candidate nodules.'.format(cluster_centroids.shape[0])
    return cluster_centroids

#获得候选结节
def get_candidate_nodule_coords():
    file_list = glob('../output/test_masks/CT_mask/*.npy')
    num = len(file_list)

    candidate_nodules = {}
    for idx, patient in enumerate(file_list):
        mask = np.load(patient)

        coords = get_coords(mask, 0.5)
        cluster_centroids = get_centroids(coords)

        split_path = patient.split('/')[-1]
        name = split_path.split('.')[0]

        mhd_path = '../data/sample_patients/test/' + name + '.mhd'
        itk_img = sitk.ReadImage(mhd_path)
        origin = np.array(itk_img.GetOrigin())
        spacing = np.array(itk_img.GetSpacing())

        #将得到的候选结节的中心坐标从“体素空间”坐标转换到“世界空间”坐标
        cluster_centroids = cluster_centroids * spacing
        cluster_centroids = cluster_centroids + origin

        candidate_nodules[name] = cluster_centroids
        print ('{}/{} have preprocessed: {}'.format(idx, num, patient))

    #保存候选结节的中心坐标
    fwrite = open('../output/test_masks/candidate_nodules_centroids.csv', 'w')
    for key in candidate_nodules.keys():
        nodules_centroid = candidate_nodules[key]
        for coord in nodules_centroid:
            fwrite.write(key + ',' + str(coord[0]) + ',' +
                         str(coord[1]) + ',' + str(coord[2]) + '\n')
    fwrite.close()


if __name__ == '__main__':
    get_candidate_nodule_coords()

