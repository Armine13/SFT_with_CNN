#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 18:36:45 2017

@author: arvardaz
"""


from scipy.misc import imread, imsave
import skimage
from glob import glob1
import numpy as np
from shutil import copyfile


indir = '/home/arvardaz/Dropbox/datasets/pillow_noise/'
imlist = glob1(indir, '*.png')

#imlist = ['1498131701.5209756_0570.png']
for imname in imlist:
    img = imread(indir+imname)
    
    for i, v in enumerate(np.linspace(0, 0.2, 20)):
        if i == 0:
            continue
        im_noise = skimage.util.random_noise(img, mode='gaussian',var=v)
        imsave(indir+imname[:-4]+'_{:02}.png'.format(i), im_noise)
        copyfile(indir+imname[:-3]+'csv', indir+imname[:-4]+'_{:02}.csv'.format(i))
        
#    break
    
        