#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:23:36 2017

@author: arvardaz
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import numpy as np

def projImage(points, fl):
    K_blender = np.array([[fl,   0.0000, 112.0000],[0.0000, fl, 112.0000],[0.0000,   0.0000,   1.0000]])
#    K_blender = np.array([[490.0000,   0.0000, 224.0000],[0.0000, 871.1111, 224.0000],[0.0000,   0.0000,   1.0000]])
    
    im = points[:,:3] / np.repeat(points[:,2].reshape(1002,1),3,axis=1)
    im = np.matmul(K_blender, im.transpose()).transpose()     
    return im

def equal_axis(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    
def plot_results(im, pred, pred_fl, gt, gt_fl, plot_n, i):

    # First subplot
    #################
    
    
    ax = fig.add_subplot(plot_n, 4,i*4+1)                    
    ax.imshow(im/255)
#    im = projImage(gt, gt_fl)
#    ax.plot(im[:,0], im[:,1], 'bx')
#    ax.set_xlim([0,224])
#    ax.set_ylim([0,224])
    ax.axis("off")
    
#    ax = fig.add_subplot(plot_n, 3,i*3+2)                    
#    ax.imshow(im/255)
##    im = projImage(gt, gt_fl)
##    ax.plot(im[:,0], im[:,1], 'bx')
#    gt_im = projImage(gt, pred_fl)
#    ax.plot(gt_im[:,0], gt_im[:,1],ls = 'none', c='#50BFE6', marker='o',markersize=5,linewidth=0.0,alpha=0.5)
#    ax.set_xlim([0,224])
#    ax.set_ylim([0,224])
#    ax.axis("off")
##    ax.set_title("id: {}".format(i))                                        

    # Second subplot
    #################
    red = projImage(gt, gt_fl)
    blue = projImage(pred, pred_fl)
    green = projImage(gt, pred_fl)
    black = projImage(pred, gt_fl)
    
    # red - blue
    ax = fig.add_subplot(plot_n, 4,i*4+2)                    
    
    ax.plot(red[:,0], red[:,1],ls = 'none', c='r', marker='o',markersize=5,linewidth=0.0,alpha=0.2)
    ax.plot(blue[:,0], blue[:,1],ls = 'none', c='royalblue', marker='o',markersize=5,linewidth=0.0,alpha=0.2)
    ax.set_xlim([0,224])
    ax.set_ylim([0,224])
    ax.invert_yaxis()
    ax.set_aspect('equal')

#################
    # red - green
    ax = fig.add_subplot(plot_n, 4,i*4+3)
    
    ax.plot(red[:,0], red[:,1],ls = 'none', c='r', marker='o',markersize=5,linewidth=0.0,alpha=0.2)
    ax.plot(green[:,0], green[:,1],ls = 'none', c='g', marker='o',markersize=5,linewidth=0.0,alpha=0.2)
    ax.set_xlim([0,224])
    ax.set_ylim([0,224])
    ax.invert_yaxis()
    ax.set_aspect('equal')
    
    #################
    # red - black
    ax = fig.add_subplot(plot_n, 4,i*4+4)
    
    ax.plot(red[:,0], red[:,1],ls = 'none', c='r', marker='o',markersize=5,linewidth=0.0,alpha=0.2)
    ax.plot(black[:,0], black[:,1],ls = 'none', c='k', marker='o',markersize=5,linewidth=0.0,alpha=0.2)
    ax.set_xlim([0,224])
    ax.set_ylim([0,224])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    
    
#    ax = fig.add_subplot(plot_n, 3,i*3+4)
#    gt_im = projImage(gt, gt_fl)
#    pred_fl_im = projImage(pred, pred_fl)
#    ax.plot(gt_im[:,0], gt_im[:,1],ls = 'none', c='#FF355E', marker='o',markersize=2,linewidth=0.0,alpha=0.3)
#    ax.plot(pred_fl_im[:,0], pred_fl_im[:,1],ls = 'none', c='#50BFE6', marker='o',markersize=2,linewidth=0.0,alpha=0.3)
#    ax.set_xlim([0,224])
#    ax.set_ylim([0,224])
#    ax.axis("off")
#    ax.set_aspect('equal')

#    ax.set_aspect('equal')
#    z = pred[:,2]
#    colors= (z-np.min(z))/np.max(z-np.min(z))
#    ax.scatter(pred[:,0], pred[:,1], pred[:,2],s=ps,c=colors,cmap='winter',linewidth=0.01,alpha=0.5)
#    ax.axes.get_xaxis().set_ticks([])
#    ax.axes.get_yaxis().set_ticks([])
#    ax.set_zticks([])
#    equal_axis(ax, pred[:,0], pred[:,1], pred[:,2])
#    ax.axis('off')
     
#    ax = fig.add_subplot(1, 3, 3, projection='3d')
#    ax.set_aspect('equal')
##    ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='b')
#    ax.scatter(pred[:,0], pred[:,1], pred[:,2],c='r')
#    ax.scatter(0, 0, 0, c='c', marker='o')
#    equal_axis(ax, pred[:,0], pred[:,1], pred[:,2])
    
#    plt.suptitle("id={}  RMSE = {:.4}({:.4}%)  iso loss={:.4}({:.4}%)".format(id, loss,loss_p, iso_loss, iso_loss_p),fontsize=18)
    
    



if __name__ == '__main__':
#    all_data = np.load('results/rig_easy.npz')
    all_data = np.load('results/def_hard.npz')
#    all_data = np.load('results/rig_gt.npz')
#    all_data = np.load('results/rig_hard.npz')
#    all_data = np.load('results/def_easy_.npz')
#    data = np.load('results/test45.329867.npz')
#    data = np.load('results/test_bg_best_1.56.npz')
#    data = np.load('results/test1491916305.62.npz') #latest test results
#    data = np.load('results/test1491911352.27.npz')
#    data = np.load('results/test1491399623.71.npz')
#data = np.load('results/test_p+fl_conv3_x60.npz')#
#    data = np.load('results/test_p+fl+l_conv3.npz')#    
                  
    outdir = '/home/arvardaz/Dropbox/out/'
    
    edges = np.genfromtxt("edges.csv", dtype=np.int32)
#    synth_gt_dist = np.genfromtxt("dist_norm.csv")
    

    ths = np.mean(all_data['fl_re']) + 2*np.std(all_data['fl_re'])
    ths_p = np.mean(all_data['rmse']) + 2*np.std(all_data['rmse'])
    
    #separate good cases
    data = {}
    for k in all_data.keys():
        data[k] = all_data[k][all_data['rmse'] < ths_p]
#        data[k] = all_data[k][(all_data['fl_re'] < ths)]

#    for k in all_data.keys():
#        data[k] = all_data[k][1:]
#    obj_size = 17.1
    
    plot_n = 1
    
    n = data['pred'].shape[0]

    cumul_loss = 0
    cumul_iso_loss = 0
    for k in range(1):
        fig = plt.figure(frameon=False)
        fig.set_size_inches(30,10)# 40)
        for i in range(plot_n):
            j = plot_n*k + i
            j = 0
            pred = data['pred'][j,:].reshape((1002,3))
            gt = data['gt'][j].reshape((1002,3))
            gt_fl = data['gt_fl'][j]
            pred_fl = data['pred_fl'][j]
            im = data['im'][j]
            plot_results(im, pred, pred_fl, gt, gt_fl, plot_n, i)
#        plt.show()
        plt.savefig(outdir+"results_rig_easy_fl.png".format(k), bbox_inches='tight')
