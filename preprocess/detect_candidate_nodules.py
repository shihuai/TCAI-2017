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


#获取阈值大于255.的位置的点作为结节内部的点
def get_coords(mask, threshold):

    print mask.shape
    coords = []
    index_list = np.where(mask >= 255.)
    first_axis = index_list[0]

    for idx, i in enumerate(first_axis):
        coords.append(np.array([index_list[2][idx], index_list[1][idx], index_list[0][idx]]))

    #for z in range(mask.shape[0]):
    #    for y in range(mask.shape[1]):
    #        for x in range(mask.shape[2]):
    #            if (mask[z, y, x] == 255.):
    #                coords.append(np.array([x, y, z]))

    coords = np.array(coords)
    #print 'candidate coords shape: ', coords.shape

    return coords

#计算两个空间坐标之间的欧式距离
def euclidean_dist(point1, point2):
    A = point1 - point2
    B = A * A

    return math.sqrt(sum(B))

#对候选结节内部的点进行聚类，如果两两位置间的距离小于30.0mm，
#即这两个点位于同一个结节内
def cluster_coords(coords, visited):
    count = 1

    for k in range(coords.shape[0]):
        if not visited[k]:
            visited[k] = count

            for idx in range(k + 1, coords.shape[0]):
                if visited[idx] == 0 and euclidean_dist(coords[idx, :], coords[k, :]) <= 50.0:
                    visited[idx] = count

            count += 1

    #计算候选结节的中心位置坐标
    cluster_centroids = []
    cluster_num = []
    for label in range(1, count):
        centroid = [0.0, 0.0, 0.0]
        num = 0
        for idx in range(coords.shape[0]):
            if visited[idx] == label:
                centroid[0] += coords[idx, 0]
                centroid[1] += coords[idx, 1]
                centroid[2] += coords[idx, 2]
                num += 1

        centroid[0] /= num
        centroid[1] /= num
        centroid[2] /= num

        cluster_num.append(num)
        cluster_centroids.append(centroid)

    cluster_centroids = np.array(cluster_centroids)

    print 'have %d clusters.' % (count - 1)

    return cluster_centroids, cluster_num, visited

#过滤掉小的候选结节
def filter_centroids(coords, cluster_centroids, cluster_num, visited):
    new_cluster_centroids = []

    for label in range(1, len(cluster_num)):
        count = 0

        for idx in range(coords.shape[0]):
            if visited[idx] == label and  \
                    euclidean_dist(coords[idx], cluster_centroids[label - 1]) >= 0.5:
                count += 1

        if count >= cluster_num[label - 1] / 3:
            new_cluster_centroids.append(cluster_centroids[label - 1])

    new_cluster_centroids = np.array(new_cluster_centroids)

    print 'After filter, the rest cluster centroid is: ', new_cluster_centroids.shape

    return new_cluster_centroids

#获得候选结节
def get_candidate_nodule_coords():
    file_list = glob('./output/test_masks/CT_mask/*.npy')
    num = len(file_list)

    candidate_nodules = {}
    for idx, patient in enumerate(file_list):
        mask = np.load(patient)

        coords = get_coords(mask, 0.5)
        visited = np.zeros(coords.shape[0])
        cluster_centroids, cluster_num, visited = cluster_coords(coords, visited)

        #cluster_centroids = filter_centroids(coords, cluster_centroids, cluster_num, visited)

        split_path = patient.split('/')[-1]
        name = split_path.split('.')[0]

        mhd_path = './data/sample_patients/val/' + name + '.mhd'
        itk_img = sitk.ReadImage(mhd_path)
        origin = np.array(itk_img.GetOrigin())
        spacing = np.array(itk_img.GetSpacing())

        #将得到的候选结节的中心坐标从“体素空间”坐标转换到“世界空间”坐标
        cluster_centroids = cluster_centroids * spacing
        cluster_centroids = cluster_centroids + origin

        candidate_nodules[name] = cluster_centroids
        print ('{}/{} have preprocessed: {}'.format(idx, num, patient))

    #保存候选结节的中心坐标
    fwrite = open('./output/test_masks/candidate_nodules_centroids.csv', 'w')
    for key in candidate_nodules.keys():
        nodules_centroid = candidate_nodules[key]
        for coord in nodules_centroid:
            fwrite.write(key + ',' + str(coord[0]) + ',' +
                         str(coord[1]) + ',' + str(coord[2]) + '\n')
    fwrite.close()


if __name__ == '__main__':
    get_candidate_nodule_coords()

