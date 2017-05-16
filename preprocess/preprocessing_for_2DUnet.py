#!/usr/bin/env python
# encoding: utf-8


"""
@version: python 2.7
@author: Sober.JChen
@license: Apache Licence
@contact: jzcjedu@foxmail.com
@software: PyCharm
@file: preprocessing_for_2DUnet.py
@time: 2017/3/31 12:24
"""
from __future__ import print_function, division
import SimpleITK as sitk
import math
import scipy.ndimage
import numpy as np
import csv
import cv2
import os
from glob import glob
import pandas as pd
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('tqdm 是一个轻量级的进度条小包。。。')
    tqdm = lambda x : x


# workspace是all_patients的上一级目录，需要在这个目录下新建一个npy_dir文件夹存放npy
workspace = "./"
###################################################################################

class Sober_luna16(object):
    def __init__(self, workspace):
        """param: workspace: all_patients的父目录"""
        self.workspace = workspace
        self.all_patients_path = os.path.join(self.workspace,"sample_patients/")
        self.tmp_workspace = os.path.join(self.workspace,"slices_masks/")
        self.ls_all_patients = glob(self.all_patients_path + "*.mhd")
        self.df_annotations = pd.read_csv(self.workspace + "csv_files/annotations.csv")
        self.df_annotations["file"] = self.df_annotations["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
        self.df_annotations = self.df_annotations.dropna()
        #---各种预定义
    def normalize(self,image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
        """数据标准化"""
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image
        #---数据标准化
    def set_window_width(self,image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
        """设置窗宽"""
        image[image > MAX_BOUND] = MAX_BOUND
        image[image < MIN_BOUND] = MIN_BOUND
        return image
        #---设置窗宽
    def get_filename(self,file_list, case):
        for f in file_list:
            if case in f:
                return (f)

    def make_mask(self,center,diam,z,width,height,spacing,origin):
        '''
        Center : centers of circles px -- list of coordinates x,y,z
        标注的结节的位置中心，是一个包含 x y z 的坐标
        diam : diameters of circles px -- diameter
        医生给出的结节直径
        widthXheight : pixel dim of image
        CT的长和宽，一般是512x512
        spacing = mm/px conversion rate np array x,y,z
        坐标中每个单位对应实际中的长度（单位为mm）
        origin = x,y,z mm np.array
        病人CT定义的坐标
        z = z position of slice in world coordinates mm
        z轴在真实世界中的位置，单位为mm
        '''
        mask = np.zeros([height,width])
        # mask中除了结节的区域，其他都是0
        # 从世界坐标装换为体素空间
        # 定义结节所在的体素范围
        v_center = (center-origin)/spacing
        v_diam = int(diam/spacing[0]+5)
        v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
        v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
        v_ymin = np.max([0,int(v_center[1]-v_diam)-5])
        v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])
        v_xrange = range(v_xmin,v_xmax+1)
        v_yrange = range(v_ymin,v_ymax+1)
        # Convert back to world coordinates for distance calculation
        # Fill in 1 within sphere around nodule
        for v_x in v_xrange:
            for v_y in v_yrange:
                p_x = spacing[0]*v_x + origin[0]
                p_y = spacing[1]*v_y + origin[1]
                if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                    mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
        return(mask)


    def myselfhandler(self):
        """自己处理"""
        for fcount, img_file in enumerate(tqdm(self.ls_all_patients)):
            mini_df = self.df_annotations[self.df_annotations["file"] == img_file]  # 获取这个病人的所有结节信息
            if mini_df.shape[0] > 0:  # 有些病人可能没有结节，跳过这些病人some files may not have a nodule--skipping those
                # load the data once
                itk_img = sitk.ReadImage(img_file)
                img_array = sitk.GetArrayFromImage(itk_img)  # 索引为z（切片数量）,x（宽）,y（高）---indexes are z,y,x (notice the ordering)
                num_z, height, width = img_array.shape  # height * width constitute the transverse plane
                origin = np.array(itk_img.GetOrigin())  #  x,y,z  以世界坐标为原点时体素空间结节中心的坐标 Origin in world coordinates (mm)
                spacing = np.array(itk_img.GetSpacing())  # 在世界坐标中各个方向体素的间距. (mm)
                # go through all nodes (why just the biggest?)
                for node_idx, cur_row in mini_df.iterrows():
                    node_x = cur_row["coordX"]
                    node_y = cur_row["coordY"]
                    node_z = cur_row["coordZ"]
                    diam = cur_row["diameter_mm"]
                    # 只取了过结节中心的切片和相邻两张切片
                    # 这里原来的教程是取三张切片，gt还是用原来的直径大小；
                    # 后来我师兄whaozl发现一个问题就是有些尺寸小的结节，相邻切片压根没切到什么东西
                    # 所以后来我们改成了只取单张切片后做数据增强的方法来增加训练集
                    # slice = np.ndarray([ height, width], dtype=np.float32)
                    # nodule_masks = np.ndarray([height, width], dtype=np.uint8)
                    w_nodule_center = np.array([node_x, node_y, node_z])   # 世界空间中结节中心的坐标
                    v_nodule_center = np.rint((w_nodule_center - origin) / spacing)   # 体素空间中结节中心的坐标 (still x,y,z ordering)
                    # np.rint 对浮点数取整，但不改变浮点数类型
                    # for i, i_z in enumerate(np.arange(int(v_nodule_center[2]) - 1,int(v_nodule_center[2]) + 2).clip(0,num_z - 1)):  # clip 方法的作用是防止超出切片数量的范围
                    i_z = int(v_nodule_center[2])
                    nodule_mask = self.make_mask(w_nodule_center, diam, i_z * spacing[2] + origin[2],width, height, spacing, origin)
                    nodule_mask = scipy.ndimage.interpolation.zoom(nodule_mask, [0.5, 0.5], mode='nearest')
                    nodule_mask[nodule_mask < 0.5] = 0
                    nodule_mask[nodule_mask > 0.5] = 1
                    nodule_mask = nodule_mask.astype(np.uint8)
                    slice = img_array[i_z]
                    slice = scipy.ndimage.interpolation.zoom(slice, [0.5, 0.5], mode='nearest')
                    slice = 255.0 * self.normalize(slice)
                    slice = slice.astype(np.uint8)#---因为int16有点大，我们改成了uint8图（值域0~255）
                    np.save(os.path.join(self.tmp_workspace, "%s_%04d_%04d.npy" % (cur_row["seriesuid"],fcount, node_idx)), slice)
                    np.save(os.path.join(self.tmp_workspace, "%s_%04d_%04d_o.npy" % (cur_row["seriesuid"],fcount, node_idx)),nodule_mask)
                    #===================================
                    #---以下代码是生成图片来观察分割是否有问题的
                    nodule_mask = 255.0 * nodule_mask
                    nodule_mask = nodule_mask.astype(np.uint8)
                    cv2.imwrite(os.path.join(self.tmp_workspace, "jpg/%04d_%04d_%04d_%s.jpg" % (fcount, node_idx,i_z,cur_row["seriesuid"])),slice)
                    cv2.imwrite(os.path.join(self.tmp_workspace, "jpg/%04d_%04d_%04d_%s_o.jpg" % (fcount, node_idx,i_z,cur_row["seriesuid"])),nodule_mask)

if __name__ == '__main__':
    sl = Sober_luna16(workspace)
    sl.myselfhandler()
