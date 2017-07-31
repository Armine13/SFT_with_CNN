#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:48:35 2017

@author: arvardaz
"""

from vgg16_model import vgg16

import tensorflow as tf
import numpy as np

import os
from glob import glob1
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import time
from matplotlib import pyplot as plt

def getFileList(datapath):
    file_names_csv = glob1(datapath ,"*.csv")
    file_list_csv = [os.path.join(datapath, fname) for fname in file_names_csv]
    file_list_png = [os.path.join(datapath, fname[:-3] + 'png') for fname in file_names_csv] #remove extension
    return (file_list_png, file_list_csv)

test_weights_path = 'weights/weights_rig_all.npz'

#testpath = "/home/isit/armine/Dropbox/datasets/fl/rig_rt+fl/test/"
#testpath = "/home/isit/armine/Dropbox/datasets/fl/test/"
#testpath = "/home/isit/armine/Dropbox/datasets/fl/def_rt+fl+l+bg/test/"
testpath = "/home/isit/armine/Dropbox/datasets/fl/rig_rt+fl+l+bg/test/"

_, filenames_test = getFileList(testpath)

print_step = 1

with tf.Graph().as_default():
    
    x = tf.placeholder(tf.float32, [None,224, 224, 3])
    y = tf.placeholder(tf.float32, [None,3006])
    fl = tf.placeholder(tf.float32, shape=None)

    model = vgg16(x)
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'

    with tf.Session(config = config) as sess:
        sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()))

        model.load_retrained_weights(test_weights_path, sess)
        cost_test = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, model.pred))))
        cost_test_fl = tf.reduce_mean(tf.divide(tf.abs(fl - model.pred_fl), fl))
                
        losses = []
        losses_fl = []
        
        fname = str(time.time())
        
        n_saved = min(1000, len(filenames_test))
        data = {}
        
        
        data['pred'] = np.empty((n_saved, 3006))
        data['pred_fl'] = np.empty((n_saved))
        data['gt'] = np.empty((n_saved, 3006))
        data['gt_fl'] = np.empty((n_saved))
        data['im'] = np.empty((n_saved, 224, 224, 3))
        data['rmse'] = np.empty((n_saved))
        data['fl_re']= np.empty((n_saved))
        
        print("Testing..")
        step = 0
            
        for i in range(n_saved):
            try:
                text = np.genfromtxt(filenames_test[i], dtype=None)
                arr = str(text).split(',')
                pts = arr[2:]
                imname = arr[0]
                im_cont = tf.read_file(testpath+imname)
                
                points = np.asarray(pts, dtype=float)
                points_test = np.reshape(points, (1, 3006))
                
                fl_test = np.float32(arr[1])
                image = tf.image.decode_png(im_cont, channels=3)
                
                image.set_shape([224, 224, 3])
                
                image_test = tf.image.per_image_standardization(image)
                image_test = np.reshape(image_test.eval(),(1, 224, 224, 3))
                
                test_loss = cost_test.eval(feed_dict={x: image_test, fl: fl_test, y:points_test})
                test_fl_loss = cost_test_fl.eval(feed_dict={x: image_test, fl: fl_test, y:points_test})
                
                
                losses.append(test_loss)
                losses_fl.append(test_fl_loss)
                
                if print_step!=-1 and step % print_step == 0:
                    print('Step %d: loss = %.2f loss_fl = %.2f ' % (step, test_loss, test_fl_loss))
                if step < n_saved:
                    data['gt'][step,:] = points
                    data['gt_fl'][step] = fl_test
                    data['pred'][step,:] = model.pred.eval(feed_dict={x: image_test, fl:fl_test, y:points_test})
                    data['pred_fl'][step] = model.pred_fl.eval(feed_dict={x: image_test, fl:fl_test, y:points_test})
                    data['im'][step] = image.eval()
                    data['rmse'][step] = test_loss
                    data['fl_re'][step] = test_fl_loss
                    
                step += 1
            except tf.errors.NotFoundError as e:
                    print("corrupt file")
        mean_loss = np.mean(losses)
        mean_loss_fl = np.mean(losses_fl)
        
        
        print("Mean testing loss: {} std={} min={} max={}".format(mean_loss,np.std(losses), min(losses), max(losses)))
        print("Mean testing FL loss: {} std={} min={} max={}".format(mean_loss_fl,np.std(losses_fl), min(losses_fl), max(losses_fl)))
    
        
        np.savez('results/test'+fname+'.npz', **data)
        print("Results saved to {}.".format('results/test'+fname+'.npz'))