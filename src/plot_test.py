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

    
def plot_results(im, pred, gt, gt_fl, loss=None, loss_p=None, iso_loss=None, iso_loss_p=None, id=None):

    # First subplot
    #################
    
    
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 5,1)                    
    ax.imshow(im/255)
#    im = projImage(gt, gt_fl)
#    ax.plot(im[:,0], im[:,1], 'bx')
    ax.set_xlim([0,224])
    ax.set_ylim([0,224])
    ax.axis("off")
    
    ax = fig.add_subplot(1, 5,2)                    
    ax.imshow(im/255)
#    im = projImage(gt, gt_fl)
#    ax.plot(im[:,0], im[:,1], 'bx')
    im = projImage(pred, gt_fl)
    ax.plot(im[:,0], im[:,1],ls = 'none', c='tomato', marker='o',markersize=2)
    ax.set_xlim([0,224])
    ax.set_ylim([0,224])
    ax.axis("off")
#    ax.set_title("id: {}".format(i))                                        

    # Second subplot
    #################
    ax = fig.add_subplot(1, 5,3, projection='3d')
    ax.set_aspect('equal')
    
    
    ax.scatter(gt[:,0], gt[:,1], gt[:,2], s=10,  c='dodgerblue')

    equal_axis(ax, pred[:,0], pred[:,1], pred[:,2])
    
    ax = fig.add_subplot(1, 5,4, projection='3d')
    ax.set_aspect('equal')

    ax.scatter(pred[:,0], pred[:,1], pred[:,2],s=10, c='tomato')
#    ax.scatter(0, 0, 0, c='c', marker='o')
    equal_axis(ax, pred[:,0], pred[:,1], pred[:,2])
    
    ax = fig.add_subplot(1, 5,5, projection='3d')
    ax.set_aspect('equal')
    ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='dodgerblue', s=10)
    ax.scatter(pred[:,0], pred[:,1], pred[:,2],c='tomato',s=10)
#    ax.scatter(0, 0, 0, c='c', marker='o')
    equal_axis(ax, pred[:,0], pred[:,1], pred[:,2])
#    
#    ax = fig.add_subplot(1, 3, 3, projection='3d')
#    ax.set_aspect('equal')
##    ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='b')
#    ax.scatter(pred[:,0], pred[:,1], pred[:,2],c='r')
#    ax.scatter(0, 0, 0, c='c', marker='o')
#    equal_axis(ax, pred[:,0], pred[:,1], pred[:,2])
    
#    plt.suptitle("id={}  RMSE = {:.4}({:.4}%)  iso loss={:.4}({:.4}%)".format(id, loss,loss_p, iso_loss, iso_loss_p),fontsize=18)
    
    plt.show()
    
def plot_results2(pred, gt, gt_fl, err):

    # First subplot
    #################
    
    
    fig = plt.figure()
#    ax = fig.add_subplot(1, 5,1)                    
#    ax.imshow(im/255)
#    im = projImage(gt, gt_fl)
#    ax.plot(im[:,0], im[:,1], 'bx')
#    ax.set_xlim([0,224])
#    ax.set_ylim([0,224])
#    ax.axis("off")
    
    ax = fig.add_subplot(1, 2, 1)
##    ax.imshow(im/255)
    im = projImage(gt, gt_fl)
    ax.plot(im[:,0], im[:,1],ls='none', c='r', marker='o', markersize=2,markeredgewidth=0.0)
    im = projImage(pred, gt_fl)
    ax.plot(im[:,0], im[:,1],ls = 'none', c='b', marker='o',markersize=2,markeredgewidth=0.0)
    ax.set_xlim([0,224])
    ax.set_ylim([0,224])
    ax.set_aspect('equal')
#    ax.axis("off")
    ax.set_title("id: {}".format(i))                                        

    # Second subplot
    #################
#    ax = fig.add_subplot(1, 1,1, projection='3d')
#    ax.set_aspect('equal')
#    
#    
#    ax.scatter(gt[:,0], gt[:,1], gt[:,2], s=10,  c='dodgerblue')
#
#    equal_axis(ax, pred[:,0], pred[:,1], pred[:,2])
#    
#    ax = fig.add_subplot(1, 3, 2, projection='3d')
#    ax.set_aspect('equal')
#
#    ax.scatter(pred[:,0], pred[:,1], pred[:,2],s=10, c='tomato')
##    ax.scatter(0, 0, 0, c='c', marker='o')
#    equal_axis(ax, pred[:,0], pred[:,1], pred[:,2])
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_aspect('equal')
    ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='r', s=10, linewidth='0')
    ax.scatter(pred[:,0], pred[:,1], pred[:,2],c='b',s=10, linewidth='0')
#    ax.scatter(0, 0, 0, c='c', marker='o')
    equal_axis(ax, np.vstack((pred[:,0], gt[:,0])), np.vstack((pred[:,1], gt[:,1])), np.vstack((pred[:,2], gt[:,2])))
    ax.set_title(err)                                        
#    
#    ax = fig.add_subplot(1, 3, 3, projection='3d')
#    ax.set_aspect('equal')
##    ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='b')
#    ax.scatter(pred[:,0], pred[:,1], pred[:,2],c='r')
#    ax.scatter(0, 0, 0, c='c', marker='o')
#    equal_axis(ax, pred[:,0], pred[:,1], pred[:,2])
    
#    plt.suptitle("id={}  RMSE = {:.4}({:.4}%)  iso loss={:.4}({:.4}%)".format(id, loss,loss_p, iso_loss, iso_loss_p),fontsize=18)
    
    plt.show()



if __name__ == '__main__':
#    data = np.load('/media/arvardaz/network_drive/SFT_with_CNN/src/results/def_easy.npz')
    data = np.load('/media/arvardaz/network_drive/SFT_with_CNN/src/results/test1499853910.8.npz')
#    data = np.load('results/test45.329867.npz')
#    data = np.load('results/test_bg_best_1.56.npz')
#    data = np.load('results/test1491916305.62.npz') #latest test results
#    data = np.load('results/test1491911352.27.npz')
#    data = np.load('results/test1491399623.71.npz')
#data = np.load('results/test_p+fl_conv3_x60.npz')#
#    data = np.load('results/test_p+fl+l_conv3.npz')#    
                  
    outdir = '/home/arvardaz/SFT_with_CNN/src/figures/'
    
#    edges = np.genfromtxt("edges.csv", dtype=np.int32)
#    synth_gt_dist = np.genfromtxt("dist_norm.csv")
    
    obj_size = 17.1
    
    plot_n = 20
    
    n = data['pred'].shape[0]
    cumul_loss = 0
    cumul_iso_loss = 0
#    for k in range(n/10):

    for i in range(plot_n):
        pred = data['pred'][i,:].reshape((1002,3))
        gt = data['gt'][i].reshape((1002,3))
        gt_fl = data['gt_fl'][i]
        pred_fl = data['pred_fl'][i]
#            im = data['im'][i]
        err = np.sqrt(np.mean(np.square(gt-pred)))        
        plot_results2(pred, gt, gt_fl, err)
        print(np.sqrt(np.mean(np.square(gt-pred))))
#        loss = np.sqrt(((pred - gt) ** 2).mean())
#        synth_gt_dist = np.asarray([np.sqrt(np.sum(np.square(a-b))) for a,b in gt[edges]])
#        synth_edge_len= np.mean(synth_gt_dist)
#        pred_dist = np.asarray([np.sqrt(np.sum(np.square(a-b))) for a,b in pred[edges]])
        
#        iso_loss =np.mean(np.abs(synth_gt_dist - pred_dist))
        
#        cumul_loss += loss
#        cumul_iso_loss += iso_loss
        
#        loss = np.sqrt(np.mean(np.square(gt-pred)))
#        loss_p = loss / obj_size * 100
        
#        iso_loss =np.mean(np.abs(synth_gt_dist - pred_dist))
#        iso_loss_p = iso_loss / synth_edge_len * 100
        
        #, loss, loss_p, iso_loss, iso_loss_p, i)
#        print("{0:6}. RMSE = {1:7.4f} ({2:8.4f}%)  mean_iso_loss = {3:7.4f} ({4:8.4f}%)".format(i, loss, loss_p, iso_loss, iso_loss_p))
#    mean_loss = cumul_loss / n
#    mean_loss_p = mean_loss / obj_size * 100
#    mean_iso_loss = cumul_iso_loss / n
#    mean_iso_loss_p = mean_iso_loss / synth_edge_len * 100
#    print("-----------------------------------------------------------------------------")
#    print("Mean RMSE    : {:7.4f} ({:8.4f}%)".format(mean_loss, mean_loss_p))
#    print("Mean iso loss: {:7.4f} ({:8.4f}%)".format(mean_iso_loss, mean_iso_loss_p))
    
    
#    plt.savefig('foo.png', bbox_inches='tight')
#    plt.savefig('foo.pdf', bbox_inches='tight')
    
#    print("Mean GT edge length = {}".format(np.mean(synth_gt_dist)))
#    print("Mean predicted edge length = {}".format(np.mean(pred_dist)))
    