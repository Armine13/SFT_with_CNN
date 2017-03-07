#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:23:36 2017

@author: arvardaz
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def equal_axis(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


gt_all = np.genfromtxt('results/gt_test1488809562.44.csv')
pred_all = np.genfromtxt('results/pred_test1488809562.44.csv')

    
for i in range(gt_all.shape[0]):
    
    gt = gt_all[i].reshape((1002,3))
    
    loss = pred_all[i,0]
    pred = pred_all[i,1:].reshape((1002,3))
       
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    
    ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='b')
    ax.scatter(pred[:,0], pred[:,1], pred[:,2],c='r')
    ax.scatter(0, 0, 0, c='c', marker='o')
    ax.set_title("RMSE = {}".format(loss))
    
    equal_axis(ax, gt[:,0], gt[:,1], gt[:,2])
    
    plt.show()
