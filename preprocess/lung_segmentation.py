# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:52:14 2017

@author: banggui
"""

from full_prep import full_prep
import numpy as np
import os

if __name__ == "__main__":
    filelist = full_prep(data_path = '../data/sample_patients/test/*.mhd', prep_folder = '../data/lung_masks/test', use_existing = False)
    print filelist
    print len(filelist)
