#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 12:22:52 2017

@author: arvardaz
"""
from numpy import cross, eye, dot
from scipy.linalg import expm3, norm
def M(axis, theta):
    return expm3(cross(eye(3), axis/norm(axis)*theta))


from vgg16_model import vgg16

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imrotate
from plot import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from glob import glob1
import os
import matplotlib.image as mpimg

from tensorflow.python.framework import ops

def getFileList(datapath):
    file_names_jpg = glob1(dr,"*.JPG")+glob1(dr,"*.png")
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
#    im = np.matmul(K, im0.transpose()).transpose()     
    return im0

def plot_results(pred, gt, im=None, loss=None, loss_p=None, iso_loss=None, iso_loss_p=None, id=None):

        # First subplot
    #################
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1)                    
#    ax.imshow(im)
    
#    im = projImage(pred, K)
#    ax.plot(im[:,0], im[:,1], 'rx')
    
    im2 = projImage(gt, None)
    ax.plot(im2[:,0], im2[:,1], 'bx')
    
    # Second subplot
    #################
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.scatter(pred[:,0], pred[:,1], pred[:,2],c='r')
    ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='b')
    ax.scatter(0, 0, 0, c='c', marker='o')                
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    equal_axis(ax, np.hstack((gt[:,0], pred[:,0])), np.hstack((gt[:,1], pred[:,1])), np.hstack((gt[:,2], pred[:,2])))      
    
    plt.suptitle("id={}\n RMSE = {:.4f}({:.4f}%)\n iso loss={:.4f}({:.4f}%)".format(id, loss, loss_p, iso_loss,iso_loss_p),fontsize=18)
    
    plt.show()
    
    return fig
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
if __name__ == '__main__':
    
    dr = '/home/arvardaz/SFT_with_CNN/temp/'
    
    with tf.Graph().as_default():
        
        with tf.Session() as sess:
            imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
#            vgg = vgg16(imgs, 'weights/weights_fc_conv3_best_all_weights.npz', sess)
            obj_size = 17.1
            
            img_file, gt_file = getFileList(dr)
                    
            cumul_loss = 0
            cumul_iso_loss = 0
            for i in range(5):#range(len(img_file)):                
                img = imread(img_file[i], mode='RGB')
                
                img = imresize(img, (224, 224))
                img_tensor = ops.convert_to_tensor(img, dtype=tf.float32)
                imout = sess.run(tf.image.per_image_standardization(img_tensor))
                
                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1) 
                ax.imshow(rgb2gray(img))#, cmap=plt.get_cmap('gray'))
                
                ax = fig.add_subplot(2, 2, 2) 
                ax.imshow(rgb2gray(imout))#, cmap=plt.get_cmap('gray'))
#                gt = np.genfromtxt(gt_file[i])
#                
#                gt = gt.reshape((1002, 3))
#
#
#            
#                pred = sess.run(vgg.pred, feed_dict={vgg.imgs: [img]})
#                                
#                pred.resize((1002, 3))
#                
#                
#                loss = np.sqrt(np.mean(np.square(gt-pred)))
#                loss_p = loss / obj_size * 100
#                cumul_loss += loss
#                
#                
#                print("{:6}. {:8}  RMSE = {:7.4f} ({:8.4f}%)  mean_iso_loss = {:7.4f} ({:8.4f}%)".format(
#                        i,img_file[i][-12:], loss, loss_p))
##                if i < 5:
#                plot_results(pred, gt, img0, loss, loss_p, i)#.savefig(\
#                    #'/home/arvardaz/SFT_with_CNN/src/figures/p+fc_conv3/'+img_file[i][-12:])
#            
#            mean_loss = cumul_loss / len(img_file)
#            mean_loss_p = mean_loss / obj_size * 100
#            
#            print("-----------------------------------------------------------------------------")
#            print("Mean RMSE    : {:7.4f} ({:8.4f}%)".format(mean_loss, mean_loss_p))
#            
#    
