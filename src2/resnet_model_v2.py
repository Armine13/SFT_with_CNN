
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
from collections import namedtuple
import six
from tensorflow.python.training import moving_averages



class resnet:
    def __init__(self, imgs, batch_size, mode, weights=None, sess=None):
        self.imgs = imgs
        self.mode = mode
        
#        self.fc_layers()
#        self.pred = self.fc3l #tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_retrained_weights(weights, sess)

        
        self.train3d = True
        self.trainFl = True
        self.trainDet = True
        
        self._extra_train_ops = []
        
        self.use_bottleneck = False
        self.relu_leakiness = 0.1
        
        self.batch_size = batch_size
#        self.hps = resnet_model.HParams(batch_size=batch_size,
        self.num_classes=3006
#                             min_lrn_rate=0.0001,
#                             lrn_rate=0.01,
        self.num_residual_units=5
#                             use_bottleneck=False,
#                             weight_decay_rate=0.0002,
#                             relu_leakiness=0.1,
#                             optimizer='adam')
        with tf.variable_scope('model'):
            self._build_model()
#        self.parameters = []
        
        # zero-mean input
        #with tf.name_scope('preprocess') as scope:
        #    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        #    images = self.imgs-mean

        # conv1_1

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, trainable):
        """Convolution."""
        with tf.variable_scope(name):
          n = filter_size * filter_size * out_filters
          kernel = tf.get_variable(
              'DW', [filter_size, filter_size, in_filters, out_filters],
              tf.float32, initializer=tf.random_normal_initializer(
                  stddev=np.sqrt(2.0/n)),trainable=trainable)
          return tf.nn.conv2d(x, kernel, strides, padding='SAME')
    
    
    def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False, trainable=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
          with tf.variable_scope('shared_activation'):
            x = self._batch_norm('init_bn', x,trainable=trainable)
            x = self._relu(x, self.relu_leakiness)
            orig_x = x
        else:
          with tf.variable_scope('residual_only_activation'):
            orig_x = x
            x = self._batch_norm('init_bn', x,trainable=trainable)
            x = self._relu(x, self.relu_leakiness)
    
        with tf.variable_scope('sub1'):
          x = self._conv('conv1', x, 3, in_filter, out_filter, stride,trainable=trainable)
    
        with tf.variable_scope('sub2'):
          x = self._batch_norm('bn2', x,trainable=trainable)
          x = self._relu(x, self.relu_leakiness)
          x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1],trainable=trainable)
    
        with tf.variable_scope('sub_add'):
          if in_filter != out_filter:
            orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
            orig_x = tf.pad(
                orig_x, [[0, 0], [0, 0], [0, 0],
                         [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
          x += orig_x
    
        tf.logging.debug('image after unit %s', x.get_shape())
        return x
    
    
    def _batch_norm(self, name, x, trainable):
        """Batch normalization."""
        with tf.variable_scope(name):
          params_shape = [x.get_shape()[-1]]
    
          beta = tf.get_variable(
              'beta', params_shape, tf.float32,
              initializer=tf.constant_initializer(0.0, tf.float32),trainable=trainable)
          gamma = tf.get_variable(
              'gamma', params_shape, tf.float32,
              initializer=tf.constant_initializer(1.0, tf.float32),trainable=trainable)
    
          if self.mode == 'train':
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
    
            moving_mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            moving_variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
    
            self._extra_train_ops.append(moving_averages.assign_moving_average(
                moving_mean, mean, 0.9))
            self._extra_train_ops.append(moving_averages.assign_moving_average(
                moving_variance, variance, 0.9))
          else:
            mean = tf.get_variable(
                'moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False)
            variance = tf.get_variable(
                'moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False)
            tf.summary.histogram(mean.op.name, mean)
            tf.summary.histogram(variance.op.name, variance)
          # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
          y = tf.nn.batch_normalization(
              x, mean, variance, beta, gamma, 0.00001)
          y.set_shape(x.get_shape())
          return y
    
    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
    
    def _fully_connected(self, x, out_dim, trainable):
        """FullyConnected layer for final output."""
        x = tf.reshape(x, [self.batch_size, -1])
        w = tf.get_variable(
            'DW', [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0),trainable=trainable)
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(),trainable=trainable)
        return tf.nn.xw_plus_b(x, w, b)
    
    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
   
    def _build_model(self):
          
        self.retrained_parameters = []
        """Build the core model within the graph."""
        with tf.variable_scope('init'):
          x = self.imgs
          x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1),self.train3d)
    
        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        if self.use_bottleneck:
          res_func = self._bottleneck_residual
          filters = [16, 64, 128, 256]
        else:
          res_func = self._residual
          filters = [16, 16, 32, 64]
          # Uncomment the following codes to use w28-10 wide residual network.
          # It is more memory efficient than very deep residual network and has
          # comparably good performance.
          # https://arxiv.org/pdf/1605.07146v1.pdf
    #      filters = [16, 160, 320, 640]
          # Update hps.num_residual_units to 4
    
        with tf.variable_scope('unit_1_0'):
          x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                       activate_before_residual[0], self.train3d)
        for i in six.moves.range(1, self.num_residual_units):
          with tf.variable_scope('unit_1_%d' % i):
            x = res_func(x, filters[1], filters[1], self._stride_arr(1), False, self.train3d)
    
        with tf.variable_scope('unit_2_0'):
          x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                       activate_before_residual[1], self.train3d)
        for i in six.moves.range(1, self.num_residual_units):
          with tf.variable_scope('unit_2_%d' % i):
            x = res_func(x, filters[2], filters[2], self._stride_arr(1), False, self.train3d)
    
        with tf.variable_scope('unit_3_0'):
          x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                       activate_before_residual[2], self.train3d)
          x = tf.Print(x, [x])
        for i in six.moves.range(1, self.num_residual_units):
          with tf.variable_scope('unit_3_%d' % i):
            x = res_func(x, filters[3], filters[3], self._stride_arr(1), False, self.train3d)
    
        with tf.variable_scope('unit_last'):
          x = self._batch_norm('final_bn', x, self.train3d)
          x = self._relu(x, self.relu_leakiness)
          x = self._global_avg_pool(x)
          #Nan
    
        with tf.variable_scope('logit'):
          self.pred = self._fully_connected(x, self.num_classes, self.train3d)
    #      detected = self._fully_connected(x, 2)
    #      self.predictions = tf.nn.softmax(points)
          
        with tf.variable_scope('focal_length'):
            self.pred_fl = self._fully_connected(x, 1, self.trainFl)
            
        with tf.variable_scope('detection'):
            with tf.variable_scope('hidden'):
                d = self._fully_connected(x, 1024, self.trainDet)
            self.detected = self._fully_connected(d, 2, self.trainDet)
#            self.detected = tf.nn.softmax(det)
            
        # TODO_
#        with tf.variable_scope('costs'):
#          def rmse(a, b):
#              return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(a, b))))
    #          return tf.sqrt(tf.reduce_mean(tf.square(tf.sub(a, b)),axis=1))  
          
    #      xent = tf.nn.softmax_cross_entropy_with_logits(
#          xent = rmse(self.points, self.labels)
#          re = tf.divide(tf.abs(self.fl - self.pred_fl), self.fl)
    
#          d = tf.nn.softmax_cross_entropy_with_logits(logits=det, labels=tf.one_hot(tf.cast(self.presence, tf.int32), 2))
          
#          self.cost = tf.reduce_mean(xent, name='xent') + tf.reduce_mean(re)
          
          
    #      self.cost_det = tf.reduce_mean(d, name='xent')
          
    #      self.cost += self._decay()
    
    #      tf.reduce_mean(rmse(points, model.pred))
#          tf.summary.scalar('cost', self.cost)

            

#    def load_retrained_weights(self, weight_file, sess):
#        weights = np.load(weight_file)
#        keys = sorted(weights.keys())
#        print("Loading weights..")
#        a = 1
#        for i, k in enumerate(keys):
#            if k =='fc8_W' or k=='fc8_b':# or k=='fcfl3_W' or k=='fcfl3_b' or k=='fcfl2_W' or k=='fcfl2_b' or k=='fcfl1_W' or k=='fcfl1_b':
#                continue
##            if 'det' in k:
##                continue
#            sess.run(self.retrained_parameters[i].assign(weights[k]))
#            a += 1
#        print("Weights loaded")

    def save_weights(self, fname):
        conv_l = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3), (5, 1), (5, 2), (5, 3)]
        fc_l = [1,2,3]
        fl_l = [1,2,3]
        weights = {}
        for block, layer in conv_l:
            exec("weights['conv{}_{}_Kernel'] = self.conv{}_{}_kernel.eval()".format(block, layer, block, layer))
            exec("weights['conv{}_{}_biases'] = self.conv{}_{}_biases.eval()".format(block, layer, block, layer))
    
        for l in fc_l:
            exec("weights['fc{}_W'] = self.fc{}w.eval()".format(l, l))
            exec("weights['fc{}_b'] = self.fc{}b.eval()".format(l, l))
        
        for l in fl_l:
            exec("weights['fcfl{}_W'] = self.fcfl_{}w.eval()".format(l, l))
            exec("weights['fcfl{}_b'] = self.fcfl_{}b.eval()".format(l, l))    
           
        exec("weights['fcl_det_1W'] = self.fcdet_1w.eval()")
        exec("weights['fcl_det_1b'] = self.fcdet_1b.eval()")
        
        exec("weights['fcl_det_2W'] = self.fcdet_2w.eval()")
        exec("weights['fcl_det_2b'] = self.fcdet_2b.eval()")
        
        np.savez(fname, **weights)
            
#    def add_histogram_summary(self):
#        tf.summary.histogram("fc1_W", self.fc1w)
#        tf.summary.histogram("fc1_b", self.fc1b)
#        tf.summary.histogram("fc1_relu", self.fc1)
#        tf.summary.histogram("fc2_W", self.fc2w)
#        tf.summary.histogram("fc2_b", self.fc2b)
#        tf.summary.histogram("fc2_relu", self.fc2)
#        tf.summary.histogram("fc3_W", self.fc3w)
#        tf.summary.histogram("fc3_b", self.fc3b)
#        
#        tf.summary.histogram("fcfl_1W", self.fcfl_1w)
#        tf.summary.histogram("fcfl_1b", self.fcfl_1b)
#        tf.summary.histogram("fcfl_2W", self.fcfl_2w)
#        tf.summary.histogram("fcfl_2b", self.fcfl_2b)
#        tf.summary.histogram("fcfl_3W", self.fcfl_3w)
#        tf.summary.histogram("fcfl_3b", self.fcfl_3b)
#        
#        tf.summary.histogram("fcl_det_1W", self.fcdet_1w)
#        tf.summary.histogram("fcl_det_1b", self.fcdet_1b)
#        tf.summary.histogram("fcl_det_2W", self.fcdet_2w)
#        tf.summary.histogram("fcl_det_2b", self.fcdet_2b)
        
