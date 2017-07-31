#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:07:34 2017

@author: arvardaz
"""
import os,sys
from glob import glob1
from scipy.misc import imread, imsave
import numpy as np
from time import time

def randomSlice(im):
    
    a = np.random.choice(im.shape[0], 2, replace=False)
    b = np.random.choice(im.shape[1], 2, replace=False)
    
    imslice = im[min(a):max(a), min(b):max(b)]
    return imslice


directory = os.path.dirname(os.path.realpath(sys.argv[0]))
imdir = directory + '/SBU-RwC90/mixed/'
outdir = imdir + 'slices/'

imname_list = glob1(imdir, '*.jpg')

for i in range(5000):
    
    idx = np.random.choice(len(imname_list))
    
    im = imread(imdir + imname_list[idx])
    newim = randomSlice(im)
    imsave(outdir + str(i) + '.jpg', newim)