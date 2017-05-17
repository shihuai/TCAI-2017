# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:39:55 2017

@author: banggui
"""

import os
from glob import glob
import numpy as np
from scipy.io import loadmat
import h5py
from scipy.ndimage.interpolation import zoom
from skimage import measure
import warnings
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from multiprocessing import Pool
from functools import partial
from step1 import step1_python
import warnings
from write_mhd_file import write_mhd_file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10)
    return dilatedMask

# def savenpy(id):
id = 1

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def resample(imgs, spacing, new_spacing,order = 2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

def savenpy(id,filelist,prep_folder,data_path,use_existing=True):
    resolution = np.array([1,1,1])
    name = filelist[id]
    if use_existing:
        if os.path.exists(os.path.join(prep_folder,name+'_label.npy')) and os.path.exists(os.path.join(prep_folder,name+'_clean.npy')):
            print(name+' had been done')
            return
    try:
#        im, m1, m2, spacing = step1_python(os.path.join(data_path,name))
        im, m1, m2, spacing = step1_python(name)
        Mask = m1+m2

        newshape = np.round(np.array(Mask.shape)*spacing/resolution)
        xx,yy,zz= np.where(Mask)
        box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
        box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
        extendbox = extendbox.astype('int')



        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        im[np.isnan(im)]=-2000
        sliceim = lumTrans(im)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = sliceim*extramask>bone_thresh
        sliceim[bones] = pad_value
#        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
#        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
#                    extendbox[1,0]:extendbox[1,1],
#                    extendbox[2,0]:extendbox[2,1]]
#        sliceim = sliceim2[np.newaxis,...]

        name, ext = os.path.splitext(os.path.split(name)[1])
#        np.save(os.path.join(prep_folder, name + '.npy'), sliceim)
#        np.save(os.path.join(prep_folder,name+'_clean'),sliceim)
#        np.save(os.path.join(prep_folder,name+'_label'),np.array([[0,0,0,0]]))

#        for idx, slice in enumerate(sliceim):
#            slicename = "../output/test_masks/lung_masks/%d.jpg" % idx
#            cv2.imwrite(slicename, slice)

        #sliceim = sliceim.reshape(sliceim.shape[1], sliceim.shape[2], sliceim.shape[0])
        #write_mhd_file(os.path.join(prep_folder, name + '.mhd'), sliceim, sliceim.shape)
        np.save(os.path.join(prep_folder, name + '.npy'), sliceim)

    except:
        print('bug in '+name)
        raise
    print(name+' done')


def full_prep(data_path,prep_folder,n_worker = None,use_existing=True):
    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)


    print('starting preprocessing')
    pool = Pool(n_worker)
#    filelist = [f for f in os.listdir(data_path)]
    filelist = glob(data_path)
    partial_savenpy = partial(savenpy,filelist=filelist,prep_folder=prep_folder,
                              data_path=data_path,use_existing=use_existing)

    N = len(filelist)
    _ = pool.map(partial_savenpy,range(N))
    pool.close()
    pool.join()
    print('end preprocessing')
    return filelist

def plot_3d(image, threshold=-300):

    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:, :, ::-1]

    verts, faces = measure.marching_cubes_classic(p, threshold)

    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection = '3d')

    mesh = Poly3DCollection(verts[faces], alpha = 0.1)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

#def full_prep(data_path,prep_folder,n_worker = None,use_existing=True):
#    warnings.filterwarnings("ignore")
#    if not os.path.exists(prep_folder):
#        os.mkdir(prep_folder)
#
#
#    print('starting preprocessing')
##    filelist = [f for f in os.listdir(data_path)]
#    filelist = glob(data_path)
#    savenpy(0,filelist,prep_folder,data_path,use_existing=use_existing)
#
#    return sliceim

if __name__ == "__main__":
    full_prep(data_path = '../data/sample_patients/test/LKDS-00012.mhd', prep_folder = '../output/test_masks/lung_masks', use_existing = False)
#    plot_3d(sliceim, 0)
