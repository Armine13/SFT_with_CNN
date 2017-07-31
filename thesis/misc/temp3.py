# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:02:16 2017

@author: arvardaz
"""

import numpy as np
from scipy.misc import imread, imresize, imrotate, imsave
#from plot import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from glob import glob1
import os

if __name__ == '__main__':
    
    dr_in = '/home/arvardaz/SFT_with_CNN/american_pillow_gt_square/'
    dr_out = '/home/arvardaz/SFT_with_CNN/datasets/dataset_american_pillow_gt_square/'


    files = glob1(dr_in, '*.JPG')
    
    files = [f[:-4] for f in files]
    
    for fname in files:
        img0 = imread(dr_in+fname + '.JPG', mode='RGB')
        img = imresize(img0, (224, 224))
        imsave(dr_out+fname+'.png', img)
    
        data = np.genfromtxt(dr_in+fname+'.csv')
        data *= 4.65
        data = np.concatenate(([fname + '.png'], data))
        
        np.savetxt(dr_out+fname+'.csv', data.reshape(1, data.shape[0]), delimiter=",", fmt="%s")

        
        