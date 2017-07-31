"""In progress.
Parag K. Mital, Jan 2016.
"""
# %%
import tensorflow as tf
from collections import namedtuple
from math import sqrt

# %%
def linear(x, n_units, scope=None, stddev=0.05,
           activation=lambda x: x):
    """Fully-connected network.
    Parameters
    ----------
    x : Tensor
        Input tensor to the network.
    n_units : int
        Number of units to connect to.
    scope : str, optional
        Variable scope to use.
    stddev : float, optional
        Initialization's standard deviation.
    activation : arguments, optional
        Function which applies a nonlinearity
    Returns
    -------
    x : Tensor
        Fully-connected output.
    """
    shape = x.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        weights = tf.get_variable("Weights", [shape[1], n_units], tf.float32,
#                                  tf.random_normal_initializer(stddev=stddev))
                                 tf.contrib.layers.xavier_initializer())
        tf.summary.histogram("Weights", weights)
#        biases = tf.Variable(tf.constant(0, shape=[3006], dtype=tf.float32),
#                               trainable=True, name='biases')
        biases = tf.get_variable("biases", [n_units], tf.float32,
                                 initializer=tf.constant_initializer(0.0))
#                                  tf.random_normal_initializer(stddev=stddev))
        tf.summary.histogram("biases", biases)
        
        if activation == None:
            return tf.nn.bias_add(tf.matmul(x, weights), biases)
        return activation(tf.nn.bias_add(tf.matmul(x, weights), biases))


def conv2d(x, n_filters,
           k_h=5, k_w=5,
           stride_h=2, stride_w=2,
           stddev=0.05,
           activation=None,
           bias=True,
           padding='SAME',
           name="Conv2D"):
    """2D Convolution with options for kernel size, stride, and init deviation.

    Parameters
    ----------
    x : Tensor
        Input tensor to convolve.
    n_filters : int
        Number of filters to apply.
    k_h : int, optional
        Kernel height.
    k_w : int, optional
        Kernel width.
    stride_h : int, optional
        Stride in rows.
    stride_w : int, optional
        Stride in cols.
    stddev : float, optional
        Initialization's standard deviation.
    activation : arguments, optional
        Function which applies a nonlinearity
    padding : str, optional
        'SAME' or 'VALID'
    name : str, optional
        Variable scope to use.

    Returns
    -------
    x : Tensor
        Convolved input.
    """
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w', [k_h, k_w, x.get_shape()[-1], n_filters],
            initializer=tf.truncated_normal_initializer(stddev=stddev))
            #initializer=tf.contrib.layers.xavier_initializer())
#        tf.truncated_normal([4096, 4096],
#                                                         dtype=tf.float32,
#                                                         stddev=1e-1)
        conv = tf.nn.conv2d(
            x, w, strides=[1, stride_h, stride_w, 1], padding=padding)
        if bias:
            b = tf.get_variable(
                'b', [n_filters],
                initializer=tf.constant_initializer(0.0))
#                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.bias_add(conv, b)
        if activation:
            conv = activation(conv)
        return conv


class residual_network:
    def __init__(self,x, n_outputs,
                     activation=tf.nn.relu):
        """Builds a residual network.
        Parameters
        ----------
        x : Placeholder
            Input to the network
        n_outputs : TYPE
            Number of outputs of final softmax
        activation : Attribute, optional
            Nonlinearity to apply after each convolution
        Returns
        -------
        net : Tensor
            Description
        Raises
        ------
        ValueError
            If a 2D Tensor is input, the Tensor must be square or else
            the network can't be converted to a 4D Tensor.
        """
        # %%
        LayerBlock = namedtuple(
            'LayerBlock', ['num_repeats', 'num_filters', 'bottleneck_size'])
        blocks = [LayerBlock(3, 128, 32),
                  LayerBlock(3, 256, 64),
                  LayerBlock(3, 512, 128),
                  LayerBlock(3, 1024, 256)]
    
        # %%
        input_shape = x.get_shape().as_list()
        if len(input_shape) == 2:
            ndim = int(sqrt(input_shape[1]))
            if ndim * ndim != input_shape[1]:
                raise ValueError('input_shape should be square')
            x = tf.reshape(x, [-1, ndim, ndim, 1])
    
        # %%
        # First convolution expands to 64 channels and downsamples
        net = conv2d(x, 64, k_h=7, k_w=7,
                     name='conv1',
                     activation=activation)
    
        # %%
        # Max pool and downsampling
        net = tf.nn.max_pool(
            net, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
        # %%
        # Setup first chain of resnets
        net = conv2d(net, blocks[0].num_filters, k_h=1, k_w=1,
                     stride_h=1, stride_w=1, padding='VALID', name='conv2')
    
        # %%
        # Loop through all res blocks
        for block_i, block in enumerate(blocks):
            for repeat_i in range(block.num_repeats):
    
                name = 'block_%d/repeat_%d' % (block_i, repeat_i)
                conv = conv2d(net, block.bottleneck_size, k_h=1, k_w=1,
                              padding='VALID', stride_h=1, stride_w=1,
                              activation=activation,
                              name=name + '/conv_in')
    
                conv = conv2d(conv, block.bottleneck_size, k_h=3, k_w=3,
                              padding='SAME', stride_h=1, stride_w=1,
                              activation=activation,
                              name=name + '/conv_bottleneck')
    
                conv = conv2d(conv, block.num_filters, k_h=1, k_w=1,
                              padding='VALID', stride_h=1, stride_w=1,
                              activation=activation,
                              name=name + '/conv_out')
    
                net = conv + net
            try:
                # upscale to the next block size
                next_block = blocks[block_i + 1]
                net = conv2d(net, next_block.num_filters, k_h=1, k_w=1,
                             padding='SAME', stride_h=1, stride_w=1, bias=False,
                             name='block_%d/conv_upscale' % block_i)
            except IndexError:
                pass
    
        # %%
        
        net = tf.nn.avg_pool(net,
                             ksize=[1, net.get_shape().as_list()[1],
                                    net.get_shape().as_list()[2], 1],
                             strides=[1, 1, 1, 1], padding='VALID')
        net = tf.reshape(
            net,
            [-1, net.get_shape().as_list()[1] *
             net.get_shape().as_list()[2] *
             net.get_shape().as_list()[3]])
    
        # %% fc1
        with tf.variable_scope("Linear1") as scope:
            net = linear(net, 2000, scope=scope, activation=tf.nn.relu)

        # %% fc2
        with tf.variable_scope("Linear2") as scope:
            net = linear(net, 2000, scope=scope, activation=tf.nn.relu)
            
        # %% fc3
        with tf.variable_scope("Linear3") as scope:
            net = linear(net, n_outputs,scope=scope, activation=None)
            
        net = tf.multiply(net, 35.0)
        self.pred = net
        # %%
    #    return net
