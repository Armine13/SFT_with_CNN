from vgg16_model import vgg16

import tensorflow as tf
import numpy as np

import os
from glob import glob1
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import time
from matplotlib import pyplot as plt
from datetime import datetime

#from tensorflow.python.client import timeline


def getFileList(datapath):
    file_names_csv = glob1(datapath ,"*.csv")
    file_list_csv = [os.path.join(datapath, fname) for fname in file_names_csv]
    file_list_png = [os.path.join(datapath, fname[:-3] + 'png') for fname in file_names_csv] #remove extension
    return (file_list_png, file_list_csv)

def preprocessImage(im_tensor):
    #mean = tf.constant([69, 68, 64], shape=[1, 1, 3], name='img_mean', dtype=tf.uint8)
    #image = tf.cast(tf.cast(im_tensor,tf.int32) - tf.cast(mean,tf.int32), tf.float32)
    image = tf.image.per_image_standardization(im_tensor)
    return image

def read_files(path, filename_queue):
    textReader = tf.TextLineReader()
    _, csv_content = textReader.read(filename_queue)
    record_defaults = [tf.constant([], dtype=tf.string)] + [tf.constant([], dtype=tf.float32)]*3006
    all_data = tf.decode_csv(csv_content, record_defaults=record_defaults, field_delim=",")


    im_name = tf.string_join([path, all_data[0]],"/")
#    im_name = all_data[0]
    coords = tf.pack(all_data[1:])

    im_cont = tf.read_file(im_name)
    example = tf.image.decode_png(im_cont, channels=3)
    return example, coords

def input_pipeline(path, filenames, batch_size, num_epochs=None, read_threads=1):
    filenames_tensor = ops.convert_to_tensor(filenames, dtype=dtypes.string)
    filename_queue = tf.train.string_input_producer(filenames_tensor,num_epochs=num_epochs, shuffle=True)


    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    data_list=[(im, pts) for im, pts in [read_files(path,filename_queue) for _ in range(read_threads)]]

    [(im.set_shape([224, 224, 3]), pt.set_shape((3006,))) for (im,pt) in data_list]
    data_list = [(preprocessImage(im),pt) for (im, pt) in data_list]
    image_batch, points_batch = tf.train.shuffle_batch_join( data_list, batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)

    return image_batch, points_batch



def isometric_loss(pred, edges, dist):
    pred = tf.reshape(pred, (1002, 3))

    rand_indices = np.random.choice(range(len(dist)), size = 1000)

    s = 0
    for i in rand_indices:
        pred_dist = tf.sqrt(tf.reduce_sum(tf.square(pred[edges[i,0],:] - pred[edges[i, 1],:])))
        s += tf.abs(pred_dist - dist[i])
    return s

def saveWeights(vgg, fname, print_message = False):
    full_name = 'weights/weights_' + fname
    vgg.save_weights(full_name)
    
    if print_message:
        print('weights saved to {}.npz.'.format(full_name))

def runTest(data, cost, sess, print_step=10, saveData=False):
    images = data[0]
    points = data[1]
    losses = []
    fname = str(datetime.now())
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
#            test_loss = cost.eval(feed_dict={x: image_test, y:points_test,keep_prob: 1.0})
            test_loss = cost.eval(feed_dict={x: image_test, y:points_test})
            losses.append(test_loss)
            
            duration = time.time() - start_time
            if print_step!=-1 and step % print_step == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, test_loss, duration))
            if saveData and step < n_saved:
                data['gt'][step,:] = points_test
#                data['pred'][step,:] = vgg.pred.eval(feed_dict={x: image_test, y:points_test,keep_prob: 1.0})
                data['pred'][step,:] = vgg.pred.eval(feed_dict={x: image_test, y:points_test})
            step += 1
    except tf.errors.OutOfRangeError:
        pass
    finally:
        coord2.request_stop()
        coord2.join(threads2)
    mean_loss = np.mean(losses)
    
    
    print("Mean testing loss: {} std={} min={} max={}".format(mean_loss,np.std(losses), min(losses), max(losses)))

    if saveData:
        np.savez('results/test'+fname+'.npz', **data)
        print("Results saved to {}.".format('results/test'+fname+'.npz'))
    
    return mean_loss, np.std(losses)
           
if __name__ == '__main__':
    trainpath = "../datasets/dataset_rt+fl+l+bg/train"
    valpath = "../datasets/dataset_rt+fl+l+bg/val"#############################
    testpath = "../datasets/dataset_rt+fl+l+bg/test"
        
#    weights_path = 'weights/weights_1491387395.44.npz'#weights_1491396095.94.npz'
#    weights_path = 'weights/weights_latest.npz'#'weights/weights_fc_1491233872.99.npz'
    
#    test_weights_path = 'weights/weights_1492683884.22_best.npz'#weights_1492787682.16_best.npz'#weights_latest_main.npz'
    
    all_test_weights = glob1('weights', '*.npz')   
    
    tab = np.zeros((len(all_test_weights), 2))
    
    for idx,f in enumerate(all_test_weights):

        test_weights_path = 'weights/'+f

        
        
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                _, filenames_test = getFileList(testpath)
                images_batch_test, points_batch_test = input_pipeline(testpath, filenames_test, batch_size=1, num_epochs=1, read_threads=1)
            
            x = tf.placeholder(tf.float32, [None, 224, 224, 3])
            y = tf.placeholder(tf.float32, [None, 3006])
#            keep_prob = tf.placeholder(tf.float32)
#            vgg = vgg16(x, keep_prob)
            vgg = vgg16(x)
            config = tf.ConfigProto()
            config.gpu_options.allocator_type = 'BFC'

            with tf.Session(config = config) as sess:
                sess.run(tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()))

                vgg.load_retrained_weights(test_weights_path, sess)
                cost_test = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, vgg.pred))))
                mean, std = runTest(data=[images_batch_test, points_batch_test],cost=cost_test,sess=sess, print_step=-1, saveData=True)
                tab[idx,:] = ((mean), (std))

    for ii, f in enumerate(all_test_weights):
        print(f, tab[ii,0], tab[ii,1])
                    