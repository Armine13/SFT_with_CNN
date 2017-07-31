
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

class vgg16:
    def __init__(self, imgs, keep_prob, weights=None, sess=None):
        self.imgs = imgs
        self.keep_prob = keep_prob
        self.convlayers()
        self.fc_layers()
        self.pred = self.fc3l #tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
    
    def convlayers(self):
        self.parameters = []
        self.retrained_parameters = []
        
        trainConv = False ###########################################
        lastConv = True
        self.trainFc = True
        # zero-mean input
        #with tf.name_scope('preprocess') as scope:
        #    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        #    images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            self.conv1_1_kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainConv, name='weights')
            conv = tf.nn.conv2d(self.imgs, self.conv1_1_kernel, [1, 1, 1, 1], padding='SAME')
            self.conv1_1_biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainConv, name='biases')
            out = tf.nn.bias_add(conv, self.conv1_1_biases)
            
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.retrained_parameters += [self.conv1_1_kernel, self.conv1_1_biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            self.conv1_2_kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), trainConv, name='weights')
            conv = tf.nn.conv2d(self.conv1_1, self.conv1_2_kernel, [1, 1, 1, 1], padding='SAME')
            self.conv1_2_biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainConv, name='biases')
            out = tf.nn.bias_add(conv, self.conv1_2_biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.retrained_parameters += [self.conv1_2_kernel, self.conv1_2_biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            self.conv2_1_kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainConv, name='weights')
            conv = tf.nn.conv2d(self.pool1, self.conv2_1_kernel, [1, 1, 1, 1], padding='SAME')
            self.conv2_1_biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainConv, name='biases')
            out = tf.nn.bias_add(conv, self.conv2_1_biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.retrained_parameters += [self.conv2_1_kernel, self.conv2_1_biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            self.conv2_2_kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), trainConv, name='weights')
            conv = tf.nn.conv2d(self.conv2_1, self.conv2_2_kernel, [1, 1, 1, 1], padding='SAME')
            self.conv2_2_biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainConv, name='biases')
            out = tf.nn.bias_add(conv, self.conv2_2_biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.retrained_parameters += [self.conv2_2_kernel, self.conv2_2_biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            self.conv3_1_kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainConv, name='weights')
            conv = tf.nn.conv2d(self.pool2, self.conv3_1_kernel, [1, 1, 1, 1], padding='SAME')
            self.conv3_1_biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainConv, name='biases')
            out = tf.nn.bias_add(conv, self.conv3_1_biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.retrained_parameters += [self.conv3_1_kernel, self.conv3_1_biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            self.conv3_2_kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainConv, name='weights')
            conv = tf.nn.conv2d(self.conv3_1, self.conv3_2_kernel, [1, 1, 1, 1], padding='SAME')
            self.conv3_2_biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainConv, name='biases')
            out = tf.nn.bias_add(conv, self.conv3_2_biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.retrained_parameters += [self.conv3_2_kernel, self.conv3_2_biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            self.conv3_3_kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), trainConv, name='weights')
            conv = tf.nn.conv2d(self.conv3_2, self.conv3_3_kernel, [1, 1, 1, 1], padding='SAME')
            self.conv3_3_biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainConv, name='biases')
            out = tf.nn.bias_add(conv, self.conv3_3_biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.retrained_parameters += [self.conv3_3_kernel, self.conv3_3_biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            self.conv4_1_kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainConv, name='weights')
            conv = tf.nn.conv2d(self.pool3, self.conv4_1_kernel, [1, 1, 1, 1], padding='SAME')
            self.conv4_1_biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainConv, name='biases')
            out = tf.nn.bias_add(conv, self.conv4_1_biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.retrained_parameters += [self.conv4_1_kernel, self.conv4_1_biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            self.conv4_2_kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainConv, name='weights')
            conv = tf.nn.conv2d(self.conv4_1, self.conv4_2_kernel, [1, 1, 1, 1], padding='SAME')
            self.conv4_2_biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainConv, name='biases')
            out = tf.nn.bias_add(conv, self.conv4_2_biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.retrained_parameters += [self.conv4_2_kernel, self.conv4_2_biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            self.conv4_3_kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), trainConv, name='weights')
            conv = tf.nn.conv2d(self.conv4_2, self.conv4_3_kernel, [1, 1, 1, 1], padding='SAME')
            self.conv4_3_biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainConv, name='biases')
            out = tf.nn.bias_add(conv, self.conv4_3_biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [self.conv4_3_kernel, self.conv4_3_biases]
            self.retrained_parameters += [self.conv4_3_kernel, self.conv4_3_biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            self.conv5_1_kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), lastConv, name='weights')
            conv = tf.nn.conv2d(self.pool4, self.conv5_1_kernel, [1, 1, 1, 1], padding='SAME')
            self.conv5_1_biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 lastConv, name='biases')
            out = tf.nn.bias_add(conv, self.conv5_1_biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
        #    self.parameters += [self.conv5_1_kernel, self.conv5_1_biases]
            self.retrained_parameters += [self.conv5_1_kernel, self.conv5_1_biases]
        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            self.conv5_2_kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), lastConv, name='weights')
            conv = tf.nn.conv2d(self.conv5_1, self.conv5_2_kernel, [1, 1, 1, 1], padding='SAME')
            self.conv5_2_biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 lastConv, name='biases')
            out = tf.nn.bias_add(conv, self.conv5_2_biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
         #   self.parameters += [self.conv5_2_kernel, self.conv5_2_biases]
            self.retrained_parameters += [self.conv5_2_kernel, self.conv5_2_biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            self.conv5_3_kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), lastConv, name='weights')
            conv = tf.nn.conv2d(self.conv5_2, self.conv5_3_kernel, [1, 1, 1, 1], padding='SAME')
            self.conv5_3_biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 lastConv, name='biases')
            out = tf.nn.bias_add(conv, self.conv5_3_biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
#            self.parameters += [self.conv5_3_kernel, self.conv5_3_biases]
            self.retrained_parameters += [self.conv5_3_kernel, self.conv5_3_biases]

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
                                                         stddev=1e-1), trainable=self.trainFc, name='weights')
            self.fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=self.trainFc, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, self.fc1w), self.fc1b)
            
            
            
#            tf.summary.histogram("out_fc1w", self.fc1w)
#            tf.summary.histogram("out_fc1b", self.fc1b)

            self.fc1 = tf.nn.relu(fc1l)
            
            drop_out1 = tf.nn.dropout(self.fc1, self.keep_prob)
            
            
#            self.parameters += [self.fc1w, self.fc1b]
            self.retrained_parameters += [self.fc1w, self.fc1b]


        # fc2
        with tf.name_scope('fc2') as scope:
            self.fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), trainable=self.trainFc, name='weights')
            self.fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=self.trainFc, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(drop_out1, self.fc2w), self.fc2b)

#            tf.summary.histogram("out_fc2w", self.fc2w)
#            tf.summary.histogram("out_fc2b", self.fc2b)


            self.fc2 = tf.nn.relu(fc2l)
            #self.parameters += [self.fc2w, self.fc2b]
            self.retrained_parameters += [self.fc2w, self.fc2b]


        with tf.name_scope('fc3') as scope:
            self.fc3w = tf.Variable(tf.random_normal([4096, 3006],
                                                dtype=tf.float32)*tf.sqrt(2.0/3006),
                                                trainable=self.trainFc, name='weights')
            self.fc3b = tf.Variable(tf.constant(0, shape=[3006], dtype=tf.float32),
                               trainable=self.trainFc, name='biases')
#            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc1, self.fc3w), self.fc3b)
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, self.fc3w), self.fc3b)
#            tf.summary.histogram("out_fc3w", self.fc3w)
#            tf.summary.histogram("out_fc3b", self.fc3b)

            self.retrained_parameters +=[self.fc3w, self.fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print("Loading weights..")
        for i, k in enumerate(keys):
            if i < len(self.parameters):
            #print i, k, np.shape(weights[k])
                sess.run(self.parameters[i].assign(weights[k]))
        print("Weights loaded")

    def load_retrained_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print("Loading weights..")
        a = 1
        for i, k in enumerate(keys):
            #print i, k, np.shape(weights[k])
#            if a < 2:
#                continue
            sess.run(self.retrained_parameters[i].assign(weights[k]))
            a += 1
        print("Weights loaded")
