#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:46:23 2017

@author: arvardaz
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from glob import glob1

dir='../datasets/dataset_rt+fl+l+bg/train/'
files = glob1(dir, '*.png')

mu = np.zeros((224,224,3),dtype=float)
n = len(files)


for i in range(n):
    im = imread(dir+files[i], mode='RGB')
    mu += 1.0/n * im
    
m = [mu[:,:,i].mean() for i in range(3)]
print(m)
plt.imshow(mu)



#rp+fl [68, 67, 63]
#rg+fl_l [69, 68, 64]
#+bg [96, 89, 76]