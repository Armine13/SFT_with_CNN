
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
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        
        self.convlayers()
        self.fc_layers()
#        self.pred = self.rnn2#self.fc3l #tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_retrained_weights(weights, sess)
    
    def convlayers(self):
#        self.parameters = []
        self.retrained_parameters = []
        
        trainConv = False ###########################################
        lastConv = False
        self.trainFc = False
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
#            self.parameters += [self.conv4_3_kernel, self.conv4_3_biases]
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
            #self.parameters += [self.conv5_2_kernel, self.conv5_2_biases]
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
#            self.parameters += [self.fc1w, self.fc1b]
            self.retrained_parameters += [self.fc1w, self.fc1b]


        # fc2
        with tf.name_scope('fc2') as scope:
            self.fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), trainable=self.trainFc, name='weights')
            self.fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=self.trainFc, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, self.fc2w), self.fc2b)

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
#            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, self.fc3w), self.fc3b) 333##################################################
            
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, self.fc3w), self.fc3b)
#            tf.summary.histogram("out_fc3w", self.fc3w)
#            tf.summary.histogram("out_fc3b", self.fc3b)

            self.retrained_parameters +=[self.fc3w, self.fc3b]
    
    ##############################################################################
        # fce1
        with tf.name_scope('fce1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            self.fce1w = tf.Variable(tf.truncated_normal([shape, 1024],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), trainable=True, name='weights')
            self.fce1b = tf.Variable(tf.constant(1.0, shape=[1024], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fce1l = tf.nn.bias_add(tf.matmul(pool5_flat, self.fce1w), self.fce1b)

            self.fce1 = tf.nn.relu(fce1l)
            #self.parameters += [self.fc2w, self.fc2b]
            self.retrained_parameters += [self.fce1w, self.fce1b]
        
        # fce2
        with tf.name_scope('fce2') as scope:
            shape = int(np.prod(self.pool4.get_shape()[1:]))
            self.fce2w = tf.Variable(tf.truncated_normal([shape, 256],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), trainable=True, name='weights')
            self.fce2b = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool4_flat = tf.reshape(self.pool4, [-1, shape])
            fce2l = tf.nn.bias_add(tf.matmul(pool4_flat, self.fce2w), self.fce2b)

            self.fce2 = tf.nn.relu(fce2l)
            #self.parameters += [self.fc2w, self.fc2b]
            self.retrained_parameters += [self.fce2w, self.fce2b]
        
        # fce3
        with tf.name_scope('fce3') as scope:
            shape = 1024+256
            self.fce3w = tf.Variable(tf.truncated_normal([shape, 2048],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), trainable=True, name='weights')
            self.fce3b = tf.Variable(tf.constant(1.0, shape=[2048], dtype=tf.float32),
                                 trainable=True, name='biases')
            

            fce3l = tf.nn.bias_add(tf.matmul(tf.concat(1, [self.fce1, self.fce2]), self.fce3w), self.fce3b)

            self.fce3 = tf.nn.relu(fce3l)
            
            self.retrained_parameters += [self.fce3w, self.fce3b]
        
        # fce4
        with tf.name_scope('fce4') as scope:
            
            self.fce4w = tf.Variable(tf.truncated_normal([2048, 1024],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), trainable=True, name='weights')
            self.fce4b = tf.Variable(tf.constant(1.0, shape=[1024], dtype=tf.float32),
                                 trainable=True, name='biases')
            
            fce4l = tf.nn.bias_add(tf.matmul(self.fce3, self.fce4w), self.fce4b)

            self.fce4 = tf.nn.relu(fce4l)
            self.retrained_parameters += [self.fce4w, self.fce4b]
        #### RNN
        
        with tf.name_scope('rnn1') as scope:
            
            weight_r1 = tf.Variable(tf.truncated_normal([4030, 4030],
                                                        dtype=tf.float32,
                                                        stddev=1e-1), trainable=True, name='weights')
            bias_r1 = tf.Variable(tf.constant(0, shape=[4030], dtype=tf.float32),
                                  trainable=True, name='biases')
            relu_r1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(tf.concat(1, [self.fce4, self.fc3l]), weight_r1), bias_r1))
            
            
            weight_r12 = tf.Variable(tf.truncated_normal([4030, 4030],
                                                        dtype=tf.float32,
                                                        stddev=1e-1), trainable=True, name='weights')
            bias_r12 = tf.Variable(tf.constant(0, shape=[4030], dtype=tf.float32),
                                  trainable=True, name='biases')
            
            relu_r12 = tf.nn.relu(tf.nn.bias_add(tf.matmul(relu_r1, weight_r12), bias_r12))
            
            
            weight_r2 = tf.Variable(tf.truncated_normal([4030, 3006],
                                                        dtype=tf.float32,
                                                        stddev=1e-1), trainable=True, name='weights')
            bias_r2 = tf.Variable(tf.constant(0, shape=[3006], dtype=tf.float32),
                                  trainable=True, name='biases')
            self.pred = tf.nn.bias_add(tf.matmul(relu_r12 , weight_r2), bias_r2)



    def load_retrained_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print("Loading weights..")
        a = 1
        for i, k in enumerate(keys):
            if k =='fc8_W' or k=='fc8_b':
                continue
            sess.run(self.retrained_parameters[i].assign(weights[k]))
            a += 1
        print("Weights loaded")

    def save_weights(self, fname):
        conv_l = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 3)]
        fc_l = [1,2,3]
        weights = {}
        for block, layer in conv_l:
            exec("weights['conv{}_{}_Kernel'] = self.conv{}_{}_kernel.eval()".format(block, layer, block, layer))
            exec("weights['conv{}_{}_biases'] = self.conv{}_{}_biases.eval()".format(block, layer, block, layer))
    
        for l in fc_l:
            exec("weights['fc{}_W'] = self.fc{}w.eval()".format(l, l))
            exec("weights['fc{}_b'] = self.fc{}b.eval()".format(l, l))
        
        emb_l = [1,2,3,4]
        for l in emb_l:
            exec("weights['fce{}_W'] = self.fce{}w.eval()".format(l, l))
            exec("weights['fce{}_b'] = self.fce{}b.eval()".format(l, l))
        
        
#        weights['rnn1_1W'] = self.rnn1w1.eval()
#        weights['rnn1_1b'] = self.rnn1b1.eval()
#        weights['rnn1_2W'] = self.rnn1w2.eval()
#        weights['rnn1_2b'] = self.rnn1b2.eval()
#        weights['rnn1_3W'] = self.rnn1w2.eval()
#        weights['rnn1_3b'] = self.rnn1b2.eval()
#        weights['rnn2_1W'] = self.rnn2w1.eval()
#        weights['rnn2_1b'] = self.rnn2b1.eval()
#        weights['rnn2_2W'] = self.rnn2w2.eval()
#        weights['rnn2_2b'] = self.rnn2b2.eval()
#        weights['rnn2_3W'] = self.rnn2w2.eval()
#        weights['rnn2_3b'] = self.rnn2b2.eval()
        
        np.savez(fname, **weights)
            
    def add_histogram_summary(self):
        tf.summary.histogram("fc1_W", self.fc1w)
        tf.summary.histogram("fc1_b", self.fc1b)
        tf.summary.histogram("fc1_relu", self.fc1)
        tf.summary.histogram("fc2_W", self.fc2w)
        tf.summary.histogram("fc2_b", self.fc2b)
        tf.summary.histogram("fc2_relu", self.fc2)
        tf.summary.histogram("fc3_W", self.fc3w)
        tf.summary.histogram("fc3_b", self.fc3b)
#        tf.summary.histogram('rnn1_1W', self.rnn1w1)
#        tf.summary.histogram('rnn1_1b', self.rnn1b1)
#        tf.summary.histogram('rnn1_2W', self.rnn1w2)
#        tf.summary.histogram('rnn1_2b', self.rnn1b2)
#        tf.summary.histogram('rnn1_3W', self.rnn1w3)
#        tf.summary.histogram('rnn1_3b', self.rnn1b3)
#        tf.summary.histogram('rnn2_1W', self.rnn2w1)
#        tf.summary.histogram('rnn2_1b', self.rnn2b1)
#        tf.summary.histogram('rnn2_2W', self.rnn2w2)
#        tf.summary.histogram('rnn2_2b', self.rnn2b2)
#        tf.summary.histogram('rnn2_3W', self.rnn2w3)
#        tf.summary.histogram('rnn2_3b', self.rnn2b3)
