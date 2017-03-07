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

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'weights/vgg16_weights.npz', sess)
    vgg.load_retrained_weights('weights/weights_fc_1488882041.1.npz', sess)
    
    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))

    pred = sess.run(vgg.pred, feed_dict={vgg.imgs: [img1]})
    pred.resize((1002, 3))
    
#     pred = pred_all[i,1:].reshape((1002,3))
           
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    
    
    #pred[:,0] = pred[:,0]/pred[:,2]
    #pred[:,1] = pred[:,1]/pred[:,2]
    
    ax.scatter(pred[:,0], pred[:,1], pred[:,2],c='r')
    ax.scatter(0, 0, 0, c='c', marker='o')
    #ax.set_title("RMSE = {}".format(loss))
    
    equal_axis(ax, pred[:,0], pred[:,1], pred[:,2])
    
    plt.show()
        
    sess.close()            