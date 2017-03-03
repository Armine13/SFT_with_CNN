########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################


#weights: https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz

import tensorflow as tf
import numpy as np
#from scipy.misc import imread, imresize
#from imagenet_classes import class_names
import os
from glob import glob1
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from maxout import max_out


class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.pred = self.fc5l #tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []
        self.retrained_parameters = []
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainable=False, name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=False, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            self.fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), trainable=False, name='weights')
            self.fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=False, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, self.fc1w), self.fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [self.fc1w, self.fc1b]
#            self.retrained_parameters += [self.fc1w, self.fc1b]


        # fc2
        with tf.name_scope('fc2') as scope:
            self.fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), trainable=False, name='weights')
            self.fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=False, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, self.fc2w), self.fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [self.fc2w, self.fc2b]
#            self.retrained_parameters += [self.fc2w, self.fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            self.fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), trainable=True, name='weights')
            self.fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc3l = tf.nn.bias_add(tf.matmul(self.fc2, self.fc3w), self.fc3b)
            self.fc3 = tf.nn.relu(fc3l)
#            self.parameters += [self.fc3w, self.fc3b]
            self.retrained_parameters += [self.fc3w, self.fc3b]
            
        ###########################################################
        with tf.name_scope('fc4') as scope:
            self.fc4w = tf.Variable(tf.random_normal([1000, 2000],
                                                dtype=tf.float32)*tf.sqrt(2.0/2000),
                                                trainable=True, name='weights')
            self.fc4b = tf.Variable(tf.constant(0, shape=[2000], dtype=tf.float32),
                               trainable=True, name='biases')
            fc4l = tf.nn.bias_add(tf.matmul(self.fc3, self.fc4w), self.fc4b)
            tf.summary.histogram("out_fc4w", self.fc4w)
            tf.summary.histogram("out_fc4b", self.fc4b)
            
            self.fc4 = tf.nn.relu(fc4l)
            self.retrained_parameters +=[self.fc4w, self.fc4b]
        

        with tf.name_scope('fc5') as scope:
            self.fc5w = tf.Variable(tf.random_normal([2000, 3006],
                                                dtype=tf.float32)*tf.sqrt(2.0/3006),
                                                trainable=True, name='weights')
            self.fc5b = tf.Variable(tf.constant(0, shape=[3006], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc5l = tf.nn.bias_add(tf.matmul(self.fc4, self.fc5w), self.fc5b)
            tf.summary.histogram("out_fc5w", self.fc5w)
            tf.summary.histogram("out_fc5b", self.fc5b)
            self.retrained_parameters +=[self.fc5w, self.fc5b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print "Loading weights.."
        for i, k in enumerate(keys):
            if i < len(self.parameters):
            #print i, k, np.shape(weights[k])
                sess.run(self.parameters[i].assign(weights[k]))
        print "Weights loaded"

    def load_retrained_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print "Loading weights.."
        for i, k in enumerate(keys):
            #print i, k, np.shape(weights[k])
            sess.run(self.retrained_parameters[i].assign(weights[k]))
        print "Weights loaded"
        
def getFileList(datapath):
    file_names_csv = glob1(datapath ,"*.csv")
    file_list_csv = [os.path.join(datapath, fname) for fname in file_names_csv]
    file_list_png = [os.path.join(datapath, fname[:-3] + 'png') for fname in file_names_csv] #remove extension
    return (file_list_png, file_list_csv)

def read_files(filename_queue):
    textReader = tf.TextLineReader()
    _, csv_content = textReader.read(filename_queue)    
    record_defaults = [tf.constant([], dtype=tf.string)] + [tf.constant([], dtype=tf.float32)]*3006
    all_data = tf.decode_csv(csv_content, record_defaults=record_defaults, field_delim=",")

    im_name = all_data[0]
    coords = tf.pack(all_data[1:])
    
    im_cont = tf.read_file(im_name)
    example = tf.image.decode_png(im_cont, channels=3)
    return example, coords

def input_pipeline(filenames, batch_size, num_epochs=None):
    filenames_tensor = ops.convert_to_tensor(filenames, dtype=dtypes.string)
    filename_queue = tf.train.string_input_producer(filenames_tensor,num_epochs=num_epochs, shuffle=False)

    example, coords = read_files(filename_queue)    
    
    #define tensor shape       
    example.set_shape([224, 224, 3])
    coords.set_shape((3006,))
    
    image_batch, points_batch = batch_queue(example, coords)
        
    return image_batch, points_batch 

def batch_queue(examples, coords):
    
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    
    example_batch, coords_batch = tf.train.shuffle_batch( [examples, coords], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)   
    return example_batch, coords_batch
    
def isometric_loss(pred, edges, dist):
    pred = tf.reshape(pred, (1002, 3))
    
    rand_indices = np.random.choice(range(len(dist)), size = 1000)
    
    s = 0
    for i in rand_indices:
        pred_dist = tf.sqrt(tf.reduce_sum(tf.square(pred[edges[i,0],:] - pred[edges[i, 1],:])))
        s += tf.abs(pred_dist - dist[i])
    return s

if __name__ == '__main__':
    
    datapath = "../output"
    #params
    learning_rate = 0.001
    reg_constant = 0.01
    
    
    
#    train_it = 100
    num_epochs = 1
    batch_size = 1
    weights = {}
    retr_layers = range(3,6)
    
    
    edges = np.genfromtxt("edges.csv", dtype=np.int32)
    dist = np.genfromtxt("dist.csv")
    
    
    with tf.Graph().as_default():
        
        _, filenames = getFileList(datapath)
         
        # Divide train/test 
        idx = int(round(len(filenames) * 0.8))
        filenames_train = filenames[:idx]
        filenames_test = filenames[idx:]
        
        images_batch_train, points_batch_train = input_pipeline(filenames_train, batch_size, num_epochs)
        images_batch_test, points_batch_test = input_pipeline(filenames_test, batch_size, num_epochs)
        
        
        x = tf.placeholder(tf.float32, [None, 224, 224, 3])
        y = tf.placeholder(tf.float32, [None, 3006])

        writer = tf.summary.FileWriter("logs/", graph=tf.get_default_graph())

        vgg = vgg16(x)
        
        #Regularization term
        vars   = tf.trainable_variables()
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ])
    
        cost_train = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, vgg.pred)))) + 2*isometric_loss(vgg.pred, edges, dist)#tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(vgg.pred,(1002,3))[edges[0,0],:] - tf.reshape(vgg.pred,(1002,3))[edges[0,1],:])))# + lossL2 * reg_constant
        cost_test = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(y, vgg.pred))))
       
#        cost_train = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.sub(tf.reshape(y, (1002, 3)), tf.reshape(vgg.pred, (1002, 3)))), 1)))
#        cost_test = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(tf.sub(tf.reshape(y, (1002, 3)), tf.reshape(vgg.pred, (1002, 3)))), 1)))
        
#        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost_train)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, momentum=0.001).minimize(cost_train)
        #Summary for cost
        tf.summary.scalar("cost", cost_train)
        summary_op = tf.merge_all_summaries()
#        summary_op = tf.summary.merge()
        
        fname = "fc_weights_"+str(time.time())+".npz" #name of file containing weights
        with tf.Session() as sess:
#        sess = tf.InteractiveSession()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            
            vgg.load_weights('vgg16_weights.npz', sess)
            vgg.load_retrained_weights('fc_weights_1488558236.26.npz',sess)
            
            ## Traininng ######################################################
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    #            sess.graph.finalize()

            
            try:
                step = 0
                train_start_time = time.time()
                while not coord.should_stop():
#                    if step >= 100:
#                        coord.request_stop()
#                        coord.join(threads)
#                        break
                    
                    start_time = time.time()
                    
                    example, coords = sess.run([images_batch_train, points_batch_train])                                        
                    duration = time.time() - start_time
                                        
                     # Write the summaries and print an overview fairly often.
                    if step % 10 == 0:
#                        # Update the events file.
                        _, loss, summary_str = sess.run([optimizer, cost_train, summary_op], feed_dict={x: example, y:coords})
                        # Print status to stdout.
                        print('Step %d: loss = %.2f (%f sec)' % (step, loss, duration))
                        writer.add_summary(summary_str, step)
                        
#                        gt = coords.reshape((1002, 3))
#                        pred = vgg.pred.eval(feed_dict={x: example, y:coords}).reshape((1002,3))
#        
#
#                        ax = Axes3D(fig)
#        
#                        ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='b')
#                        ax.scatter(pred[:,0], gt[:,1], gt[:,2],c='r')
#                        plt.show()
                        

                        #Save weights after every 10 steps        
                        for l in retr_layers:
                            exec("weights['fc{}_W'] = vgg.fc{}w.eval()".format(l, l))
                            exec("weights['fc{}_b'] = vgg.fc{}b.eval()".format(l, l))
                        np.savez(fname, **weights)

                    else:
                        _, loss = sess.run([optimizer, cost_train], feed_dict={x: example, y:coords})
                    step += 1

    
            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached\n Training time: {} sec'.format(time.time()-train_start_time))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)
            gt = coords.reshape((1002, 3))
            pred = vgg.pred.eval(feed_dict={x: example, y:coords}).reshape((1002,3))

            fig = plt.figure()  
            ax = Axes3D(fig)
                
            ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='b')
            ax.scatter(pred[:,0], gt[:,1], gt[:,2],c='r')
            plt.show()
    #            
            
            #Saving the weights            
            weights = {}
            for l in retr_layers:
                exec("weights['fc{}_W'] = vgg.fc{}w.eval()".format(l, l))
                exec("weights['fc{}_b'] = vgg.fc{}b.eval()".format(l, l))
            np.savez(fname, **weights)
            print('weights saved to {}.'.format(fname))
            
            
            ## Testing ########################################################
            print("Evaluation..")
            losses = []
            
            coord2 = tf.train.Coordinator()
            threads2 = tf.train.start_queue_runners(coord=coord2, sess=sess)
            try:
                step = 0
                while not coord2.should_stop():
                    start_time = time.time()
                    example_test, coords_test = sess.run([images_batch_test, points_batch_test])
                    test_loss = cost_test.eval(feed_dict={x: example_test, y:coords_test})
                    losses.append(test_loss)
                    
                    duration = time.time() - start_time
                    if step % 10 == 0:
                        print('Step %d: loss = %.2f (%.3f sec)' % (step, test_loss, duration))

                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done testing -- epoch limit reached')
#            except Exception, e:
#                print repr(e)
            except KeyError, e:
                print 'I got a KeyError - reason "%s"' % str(e)
            finally:
                # When done, ask the threads to stop.
                coord2.request_stop()
                coord2.join(threads2)
                print("Average Error: {}".format(np.mean(losses)))
                
#                print(costmy.eval(feed_dict={x: example_test, y:coords_test}))
#                print(costmy1.eval(feed_dict={x: example_test, y:coords_test}))
#                plt.figure()
            gt = coords_test.reshape((1002, 3))
            pred = vgg.pred.eval(feed_dict={x: example_test, y:coords_test}).reshape((1002,3))

            fig = plt.figure()
            ax = Axes3D(fig)
            
            ax.scatter(gt[:,0], gt[:,1], gt[:,2],c='b')
            ax.scatter(pred[:,0], gt[:,1], gt[:,2],c='r')
            plt.show()
        
            writer.close()

#    