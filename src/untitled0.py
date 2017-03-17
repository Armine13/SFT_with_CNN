#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:49:56 2017

@author: arvardaz
"""

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
from glob import glob1
import os
        
def projImage(points):
    K_blender = np.array([[245.0000,   0.0000, 112.0000],[0.0000, 435.5555419921875, 112.0000],[0.0000,   0.0000,   1.0000]])
#    K_blender = np.array([[490.0000,   0.0000, 224.0000],[0.0000, 871.1111, 224.0000],[0.0000,   0.0000,   1.0000]])
    
    im0 = points[:,:3] / np.repeat(points[:,2].reshape(1002,1),3,axis=1)
    im = np.matmul(K_blender, im0.transpose()).transpose()     
    return im

def equal_axis(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    


if __name__ == '__main__':
    files = glob1("../output", "*.csv")    
    files = files[:10]
    edges = np.genfromtxt("edges_norm.csv", dtype=np.int32)
    synth_gt_dist = np.genfromtxt("dist_norm.csv")
    
    l = np.ones((len(files)))

    for i in range(len(files)):
        
        f = open('../output/'+files[0], 'rb')
        text = f.read()
        f.close()
        pred = np.asarray(text.split(',')[1:], dtype=float)
        pred = pred.reshape((1002,3))
        
        pred_dist = ([np.sqrt(np.sum(np.square(a-b))) for a,b in pred[edges]])
        l[i] =np.mean(np.abs(synth_gt_dist - pred_dist))
        
        
        
    