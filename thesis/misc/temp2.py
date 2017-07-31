#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:12:31 2017

@author: arvardaz
"""
import numpy as np
from matplotlib import pyplot as plt
from glob import glob1
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

path = '/home/arvardaz/SFT_with_CNN/temp/'

fnames = glob1(path, '*.csv')
for i in range(len(fnames)):
    text = np.genfromtxt(path+fnames[i], dtype=None)
    arr = str(text).split(',')
    pts = arr[1:]
    pts = np.asarray(pts, dtype=float)
    pts  = np.reshape(pts, [1002, 3])
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2])
    plt.show()