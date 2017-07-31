#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:12:02 2017

@author: arvardaz
"""
#from vgg16_model import vgg16

import numpy as np

import os
from glob import glob1
from scipy.misc import imread
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
#from tensorflow.python.client import timeline


def getFileList(datapath):
    file_names_csv = glob1(datapath ,"*.csv")
    file_list_csv = [os.path.join(datapath, fname) for fname in file_names_csv]
    file_list_png = [os.path.join(datapath, fname[:-3] + 'png') for fname in file_names_csv] #remove extension
    return (file_list_png, file_list_csv)
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
    
def plot_results(pred, gt, gt_fl, im):

    # First subplot
    #################
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1)                    
    ax.imshow(im)
#    pts_proj = projImage(gt, gt_fl)
#    ax.plot(pts_proj[:,0], pts_proj[:,1], 'bx')
    pts_proj = projImage(pred, gt_fl)
    ax.plot(pts_proj[:,0], pts_proj[:,1], 'rx')
    
    # Second subplot
    #################
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_aspect('equal')
    ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='b')
    ax.scatter(pred[:,0], pred[:,1], pred[:,2],c='r')
    ax.scatter(0, 0, 0, c='c', marker='o')
    equal_axis(ax, pred[:,0], pred[:,1], pred[:,2])
    plt.show()


testpath = '/media/arvardaz/network_drive/Dropbox/datasets/fl/def_rt+fl/test/'
filenames_test, _ = getFileList(testpath)

data = np.load('/media/arvardaz/network_drive/SFT_with_CNN/src/results/test1495616730.87.npz')
data = dict(data)
num = data['fl'].shape[0]
fnames = filenames_test[:num]
data['im'] = [imread(f, mode='RGB') for f in fnames]

edges = np.genfromtxt("/home/arvardaz/SFT_with_CNN/src/edges.csv", dtype=np.int32)

obj_size = 17.1
#obj_size = 3.68#mean side length

for i in range(5):

    pred = data['pred'][i,:].reshape((1002,3))
    gt = data['gt'][i].reshape((1002,3))
    gt_fl = data['fl'][i]
    im = data['im'][i]
    
    loss = np.sqrt(((pred - gt) ** 2).mean())
    synth_gt_dist = np.asarray([np.sqrt(np.sum(np.square(a-b))) for a,b in gt[edges]])
    synth_edge_len= np.mean(synth_gt_dist)
    pred_dist = np.asarray([np.sqrt(np.sum(np.square(a-b))) for a,b in pred[edges]])
    
    iso_loss =np.mean(np.abs(synth_gt_dist - pred_dist))
    
    loss = np.sqrt(np.mean(np.square(gt-pred)))
    loss_p = loss / obj_size * 100
    
    iso_loss =np.mean(np.abs(synth_gt_dist - pred_dist))
    iso_loss_p = iso_loss / synth_edge_len * 100
    
    plot_results(pred, gt, gt_fl, im)
    print("{0:6}. RMSE = {1:7.4f} ({2:8.4f}%)  mean_iso_loss = {3:7.4f} ({4:8.4f}%)".format(i, loss, loss_p, iso_loss, iso_loss_p))

