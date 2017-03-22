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

def projImage(points):
#    K_blender = np.array([[245.0000,   0.0000, 112.0000],[0.0000, 435.5555419921875, 112.0000],[0.0000,   0.0000,   1.0000]])
#    K_blender = np.array([[490.0000,   0.0000, 224.0000],[0.0000, 871.1111, 224.0000],[0.0000,   0.0000,   1.0000]])
    
    im = points[:,:3] / np.repeat(points[:,2].reshape(1002,1),3,axis=1)
#    im = np.matmul(K_blender, im.transpose()).transpose()     
    return im

def equal_axis(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    
def plot_results(pred, gt, im, loss=None, iso_loss=None):

    # First subplot
    #################
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1)                    
#    ax.imshow(im)
    im = projImage(gt)
    ax.plot(im[:,0], im[:,1], 'bx')
    im = projImage(pred)
    ax.plot(im[:,0], im[:,1], 'rx')
    
#    ax.set_xlim([0,224])
#    ax.set_ylim([0,224])
#    ax.set_title("id: {}".format(i))                                        

    # Second subplot
    #################
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    
    ax.set_aspect('equal')
    ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='b')
    ax.scatter(pred[:,0], pred[:,1], pred[:,2],c='r')
    ax.scatter(0, 0, 0, c='c', marker='o')
    equal_axis(ax, pred[:,0], pred[:,1], pred[:,2])
    
    plt.suptitle("RMSE = {:.4}\n isometric loss={:.4}".format(loss, iso_loss),fontsize=18)
    
    plt.show()



if __name__ == '__main__':
    
    data = np.load('results/test1490101705.55.npz')
#    data = np.load('results/test1490003177.42.npz')
    
    edges = np.genfromtxt("edges.csv", dtype=np.int32)
    synth_gt_dist = np.genfromtxt("dist_norm.csv")
    
    n = data['pred'].shape[0]
    cum_e = 0
    cum_il = 0
    for i in range(n):
#    for i in range(n):
        loss = data['pred'][i,0]
        pred = data['pred'][i,1:].reshape((1002,3))
        gt = data['gt'][i].reshape((1002,3))
        im = data['image'][i]
        
        
        pred_dist = ([np.sqrt(np.sum(np.square(a-b))) for a,b in pred[edges]])
        iso_loss =np.mean(np.abs(synth_gt_dist - pred_dist))
        
        cum_e += loss
        cum_il += iso_loss
        
        pred[:,1] = -pred[:,1]
        pred[:,2] = -pred[:,2]
        
        gt[:,1] = -gt[:,1]
        gt[:,2] = -gt[:,2]
        
        plot_results(pred, gt, im, loss, iso_loss)
        print("{}. RMSE={} mean_iso_loss={} ({:.3}%)".format(i, loss, iso_loss, iso_loss*100/np.mean(synth_gt_dist)))
    print("Mean RMSE : {}".format(cum_e*1.0/(i+1)))        
    print("Mean GT edge length = {}".format(np.mean(synth_gt_dist)))
    print("Mean predicted edge length = {}".format(np.mean(pred_dist)))
    print("Mean iso loss: {} ({:.3}%)".format(cum_il*1.0/(i+1), cum_il*1.0/(i+1)*100/np.mean(synth_gt_dist)))