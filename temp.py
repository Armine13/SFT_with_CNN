#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:24:13 2017

@author: arvardaz
"""

from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

k = loadmat('/home/arvardaz/SFT_with_CNN/3D_models/paper/R-Pa-IsoSm-RcRg-001_shape1/K_3D_GT.mat')

l = loadmat('/home/arvardaz/SFT_with_CNN/3D_models/paper/R-Pa-IsoSm-RcRg-001_shape1/R1_SHAPE1_KeyPoints_GT.mat')
points = l['KeyPoints'][0][0][2]


fig = plt.figure(frameon=False)
ax = fig.gca(projection='3d')

ax.scatter(points[0,:],points[1,:],points[2,:],'r.')

#plt.plot(points[0,:],points[1,:],points[2,:],'r*')
