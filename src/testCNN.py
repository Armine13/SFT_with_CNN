#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:02:16 2017

@author: arvardaz
"""

from vgg16_model import vgg16

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from plot import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from glob import glob1
import os

def getFileList(datapath):
    file_names_csv = glob1(datapath ,"*.csv")
    file_list_csv = [os.path.join(datapath, fname) for fname in file_names_csv]
    file_list_png = [os.path.join(datapath, fname[:-3] + 'JPG') for fname in file_names_csv] #remove extension
    return (file_list_png, file_list_csv)
def equal_axis(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'weights/vgg16_weights.npz', sess)
    vgg.load_retrained_weights('weights/weights_fc_1488894639.52 (another copy).npz', sess)
    
    img_file, gt_file = getFileList('/home/arvardaz/SFT_with_CNN/american_pillow_gt/')
    
    i = 0
    for i in range(len(img_file)):
        
        if i < 1:
            i += 1
        else:
            break
        
        img = imread(img_file[i], mode='RGB')
        gt = np.genfromtxt(gt_file[i])
        
        gt = gt.reshape((1002, 3))
        
        img = imresize(img, (224, 224))
    
        pred = sess.run(vgg.pred, feed_dict={vgg.imgs: [img]})
        pred.resize((1002, 3))
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal')
        
        ax.scatter(pred[:,0], pred[:,1], pred[:,2],c='r')
        ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='b')
        ax.scatter(0, 0, 0, c='c', marker='o')
        #ax.set_title("RMSE = {}".format(loss))
        
        equal_axis(ax, pred[:,0], pred[:,1], pred[:,2])
        
        plt.show()
            
    sess.close()            