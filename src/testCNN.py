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
    file_names_jpg = glob1(datapath ,"*.JPG")
    file_list_jpg = [os.path.join(datapath, fname) for fname in file_names_jpg]
    file_list_csv = [os.path.join(datapath, fname[:-3] + 'csv') for fname in file_names_jpg] #remove extension
    return (file_list_jpg, file_list_csv)
def equal_axis(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def projImage(points, K):
    im0 = points[:,:3] / np.repeat(points[:,2].reshape(1002,1),3,axis=1)
    im = np.matmul(K, im0.transpose()).transpose()     
    return im


if __name__ == '__main__':
    
    dr = '/home/arvardaz/SFT_with_CNN/american_pillow_gt/'
    K = np.array([[1887.3979937413362, 0, 1187.4168448401772],[0,1887.3979937413362,807.75879695084984],[0,0,1]])
    
    
    with tf.Graph().as_default():
        
        with tf.Session() as sess:
            imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
            vgg = vgg16(imgs, 'weights/vgg16_weights.npz', sess)
            
            vgg.load_retrained_weights('weights/weights_trained_on_dec_norm (copy).npz',sess)
#            vgg.load_retrained_weights('weights/weights_fc_1489578184.74.npz',sess)
            
            img_file, gt_file = getFileList(dr)
                    
            cum_e = 0
            for i in range(len(img_file)):                
                img = imread(img_file[i], mode='RGB')
                gt = np.genfromtxt(gt_file[i])
                
                gt = gt.reshape((1002, 3))
                
                img = imresize(img, (224, 224))
            
                pred = sess.run(vgg.pred, feed_dict={vgg.imgs: [img]})
                pred.resize((1002, 3))
                pred[:,1] = -pred[:,1]
                pred[:,2] = -pred[:,2]
                
                e = np.sqrt(np.mean(np.square(gt-pred)))
                cum_e += e
                print("{} RMSE = {}".format(i, e))
                
                if i < 10:
                    fig = plt.figure()
                    ax = fig.gca(projection='3d')
                    ax.set_aspect('equal')
                    ax.scatter(pred[:,0], pred[:,1], pred[:,2],c='r')
                    ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='b')
                    ax.scatter(0, 0, 0, c='c', marker='o')                
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title("{}  RMSE={}".format(img_file[i][-12:], e))
                    equal_axis(ax, np.hstack((gt[:,0], pred[:,0])), np.hstack((gt[:,1], pred[:,1])), np.hstack((gt[:,2], pred[:,2])))      
                    plt.show()
                    
                    fig, ax = plt.subplots()
                    ax.imshow(plt.imread(img_file[i]))
                    im = projImage(pred, K)
                    ax.plot(im[:,0], im[:,1], 'rx')
                    ax.set_title("{}  RMSE={}".format(img_file[i][-12:], e))

                
                
            print("Mean : {}".format(cum_e*1.0/len(img_file)))    
