#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:05:55 2017

@author: arvardaz
"""
from glob import glob1
import numpy as np
from scipy.misc import imread, imsave, imresize
from matplotlib import pyplot as plt

maskdir = '/home/arvardaz/SFT_with_CNN/photoscan/PILLOW MASKS/'
imdir = '/home/arvardaz/SFT_with_CNN/american_pillow_gt_def/'

#outdir = '/home/arvardaz/SFT_with_CNN/american_pillow_gt_square1/'
outdir = '/home/arvardaz/SFT_with_CNN/american_pillow_gt_def_square/'
outdirseg = '/home/arvardaz/SFT_with_CNN/american_pillow_gt_square1/segmented/'

imfiles = glob1(imdir, '*.png')

for i in range(len(imfiles)):
    name = imfiles[i]
    im = imread(imdir + name)
#    mask = imread(maskdir+imfiles[i][:-4]+'_mask.png')
    
#    segim = 64*np.ones_like(im)
#    segim[mask==255] = im[mask==255]
#    plt.imshow(newim)
    imout = im[:,420:-420]
    imout = imresize(imout,(224, 224))
    imsave(outdir+name, imout)
#    imsave(outdirseg+name, segim[:,400:-400])


