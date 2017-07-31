from vgg16_model import vgg16

import tensorflow as tf
import numpy as np

import os
from glob import glob1
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import time



def getFileList(datapath):
    file_names_csv = glob1(datapath ,"*.csv")
    file_list_csv = [os.path.join(datapath, fname) for fname in file_names_csv]
    file_list_png = [os.path.join(datapath, fname[:-3] + 'png') for fname in file_names_csv] #remove extension
    return (file_list_png, file_list_csv)

def preprocessImage(im_tensor):
    #mean = tf.constant([69, 68, 64], shape=[1, 1, 3], name='img_mean', dtype=tf.uint8)
    #image = tf.cast(tf.cast(im_tensor,tf.int32) - tf.cast(mean,tf.int32), tf.float32)
    image = tf.image.per_image_whitening(im_tensor)
    return image

def runTest(data, cost, sess, print_step=10, saveData=False):
    images = data[0]
    points = data[1]
    losses = []
    fname = str(time.time())
    coord2 = tf.train.Coordinator()
    threads2 = tf.train.start_queue_runners(coord=coord2, sess=sess)
    n_saved = 60
    data = {}
    data['pred'] = np.empty((n_saved, 3006))
    data['gt'] = np.empty((n_saved, 3006))
    try:
        print("Testing..")
        step = 0
        while not coord2.should_stop():
            
            start_time = time.time()
            image_test, points_test = sess.run([images, points])
            test_loss = cost.eval(feed_dict={x: image_test, y:points_test,keep_prob: 1.0})
            losses.append(test_loss)
            
            duration = time.time() - start_time
            if print_step!=-1 and step % print_step == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, test_loss, duration))
            if saveData and step < n_saved:
                data['gt'][step,:] = points_test
                data['pred'][step,:] = vgg.pred.eval(feed_dict={x: image_test, y:points_test,keep_prob: 1.0})
            step += 1
    except tf.errors.OutOfRangeError:
        pass
    finally:
        coord2.request_stop()
        coord2.join(threads2)
    mean_loss = np.mean(losses)
    print("Mean testing loss: {} min={} max={}".format(mean_loss, min(losses), max(losses)))

    if saveData:
        np.savez('results/test'+fname+'.npz', **data)
        print("Results saved to {}.".format('results/test'+fname+'.npz'))
    
    return mean_loss        

            
if __name__ == '__main__':
        
#    weights_path = 'weights/weights_1491387395.44.npz'#weights_1491396095.94.npz'
#    weights_path = 'weights/weights_latest.npz'#'weights/weights_fc_1491233872.99.npz'
    test_weights_path = 'weights/weights_latest_main_11_04_2.npz'
    testpath = '/home/arvardaz/SFT_with_CNN/temp2/'

    
    with tf.Graph().as_default():
        _, filenames_test = getFileList(testpath)
        images_batch_test, points_batch_test = input_pipeline(testpath, filenames_test, batch_size=1, num_epochs=1, read_threads=1)
        
        x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        y = tf.placeholder(tf.float32, [None, 3006])
        keep_prob = tf.placeholder(tf.float32)
        vgg = vgg16(x, keep_prob)
        config = tf.ConfigProto()
        config.gpu_options.allocator_type = 'BFC'

        with tf.Session(config = config) as sess:
            sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()))

            vgg.load_retrained_weights(test_weights_path, sess)
            cost_test = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, vgg.pred))))
            runTest(data=[images_batch_test, points_batch_test],cost=cost_test,sess=sess, print_step=10, saveData=False)


##!/usr/bin/env python2
## -*- coding: utf-8 -*-
#"""
#Created on Tue Mar  7 12:02:16 2017
#
#@author: arvardaz
#"""
#
#from vgg16_model import vgg16
#
#import tensorflow as tf
#import numpy as np
#from scipy.misc import imread, imresize, imrotate
##from plot import *
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#import matplotlib.pyplot as plt
#from glob import glob1
#import os
#
#def getFileList(datapath):
#    file_names_jpg = glob1(dr,"*.JPG")+glob1(dr,"*.png")
#    file_list_jpg = [os.path.join(datapath, fname) for fname in file_names_jpg]
#    file_list_csv = [os.path.join(datapath, fname[:-3] + 'csv') for fname in file_names_jpg] #remove extension
#    return (file_list_jpg, file_list_csv)
#def equal_axis(ax, X, Y, Z):
#    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
#    
#    mid_x = (X.max()+X.min()) * 0.5
#    mid_y = (Y.max()+Y.min()) * 0.5
#    mid_z = (Z.max()+Z.min()) * 0.5
#    ax.set_xlim(mid_x - max_range, mid_x + max_range)
#    ax.set_ylim(mid_y - max_range, mid_y + max_range)
#    ax.set_zlim(mid_z - max_range, mid_z + max_range)
#
#def projImage(points, K):
#    im0 = points[:,:3] / np.repeat(points[:,2].reshape(1002,1),3,axis=1)
#    im = np.matmul(K, im0.transpose()).transpose()     
#    return im
#
#def plot_results(pred, gt, im, loss=None, loss_p=None, iso_loss=None, iso_loss_p=None, id=None):
#
#        # First subplot
#    #################
#    fig = plt.figure(figsize=plt.figaspect(0.5))
#    ax = fig.add_subplot(1, 2, 1)                    
#    ax.imshow(im)
#    
#    im = projImage(pred, K)
#    ax.plot(im[:,0], im[:,1], 'rx')
#    
#    im2 = projImage(gt, K)
#    ax.plot(im2[:,0], im2[:,1], 'bx')
#    
#    # Second subplot
#    #################
#    ax = fig.add_subplot(1, 2, 2, projection='3d')
#    
#    ax = fig.gca(projection='3d')
#    ax.set_aspect('equal')
#    ax.scatter(pred[:,0], pred[:,1], pred[:,2],c='r')
#    ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='b')
#    ax.scatter(0, 0, 0, c='c', marker='o')                
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    ax.set_zlabel('Z')
#    equal_axis(ax, np.hstack((gt[:,0], pred[:,0])), np.hstack((gt[:,1], pred[:,1])), np.hstack((gt[:,2], pred[:,2])))      
#    
#    plt.suptitle("id={}\n RMSE = {:.4f}({:.4f}%)\n iso loss={:.4f}({:.4f}%)".format(id, loss, loss_p, iso_loss,iso_loss_p),fontsize=18)
#    
#    plt.show()
#    
#    return fig
#
#if __name__ == '__main__':
#    
##    dr = '/home/arvardaz/SFT_with_CNN/american_pillow_gt_square/segmented/'
#    dr = '/home/arvardaz/SFT_with_CNN/temp2/'
##    dr = '/home/arvardaz/SFT_with_CNN/american_pillow_gt_square/'
#    K = np.array([[1887.3979937413362, 0, 1187.4168448401772   -400   ],[0,1887.3979937413362,807.75879695084984],[0,0,1]])
#    
#    edges = np.genfromtxt("edges.csv", dtype=np.int32)
#    real_gt_dist = np.genfromtxt("dist_real_gt.csv")
#    
#    real_edge_len = np.mean(real_gt_dist)
#    
#    obj_size = 3.68#mean side length
#    
#    with tf.Graph().as_default():
#        
#        with tf.Session() as sess:
#            imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
#            keep_prob = tf.placeholder_with_default(1.0, [])
#
#            vgg = vgg16(imgs, keep_prob, 'weights/weights_latest_main_11_04_2.npz', sess)
##            vgg = vgg16(imgs, keep_prob, 'weights/weights_fc_conv3_best_all_weights.npz', sess)
#
#
#            img_file, gt_file = getFileList(dr)
#            
#                    
#            cumul_loss = 0
#            cumul_iso_loss = 0
#            for i in range(len(img_file)):                
#                img0 = imread(img_file[i], mode='RGB')
##                img0 = imrotate(img0, -90)
#                
#                img = imresize(img0, (224, 224))
#                
##                gt = np.genfromtxt(gt_file[i])
#                gt = np.genfromtxt(gt_file[i], delimiter=',')
#                
#                gt = gt.reshape((1002, 3))
#
#
#            
#                pred = sess.run(vgg.pred, feed_dict={vgg.imgs: [img]})
##                pred /= 4.65
#                
#                pred.resize((1002, 3))
#                #############################################
##                if pred[0,2] < 0:
##                    pred[:,1] = -pred[:,1]
##                    pred[:,2] = -pred[:,2]
#                #############################################
#                
#                pred_dist = [np.sqrt(np.sum(np.square(a-b))) for a,b in pred[edges]]
#                
#                loss = np.sqrt(np.mean(np.square(gt-pred)))
#                loss_p = loss / obj_size * 100
#                cumul_loss += loss
#                iso_loss =np.mean(np.abs(real_gt_dist - pred_dist))
#                iso_loss_p = iso_loss / real_edge_len * 100
#                cumul_iso_loss += iso_loss
#                
#                print("{:6}. {:8}  RMSE = {:7.4f} ({:8.4f}%)  mean_iso_loss = {:7.4f} ({:8.4f}%)".format(
#                        i,img_file[i][-12:], loss, loss_p, iso_loss, iso_loss_p))
##                if i < 5:
#                plot_results(pred, gt, img0, loss, loss_p, iso_loss, iso_loss_p, i)#.savefig(\
#                    #'/home/arvardaz/SFT_with_CNN/src/figures/p+fc_conv3/'+img_file[i][-12:])
#            
#            mean_loss = cumul_loss / i#len(img_file)
#            mean_loss_p = mean_loss / obj_size * 100
#            mean_iso_loss = cumul_iso_loss / i#len(img_file)
#            mean_iso_loss_p = mean_iso_loss / real_edge_len * 100
#            print("-----------------------------------------------------------------------------")
#            print("Mean RMSE    : {:7.4f} ({:8.4f}%)".format(mean_loss, mean_loss_p))
#            print("Mean iso loss: {:7.4f} ({:8.4f}%)".format(mean_iso_loss, mean_iso_loss_p))
#            
##            print("Mean RMSE : {}".format(cumul_loss*1.0/len(img_file)))
##            print("Mean GT edge length = {}".format(np.mean(real_gt_dist)))
##            print("Mean predicted edge length = {}".format(np.mean(pred_dist)))            
##            print("Mean mean iso loss: {}".format(cumul_iso_loss*1.0/len(img_file)))
#            
#    
