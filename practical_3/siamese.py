from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import initializers


class Siamese(object):
    """
    This class implements a siamese convolutional neural network in
    TensorFlow. Term siamese is used to refer to architectures which
    incorporate two branches of convolutional networks parametrized
    identically (i.e. weights are shared). These graphs accept two
    input tensors and a label in general.
    """

    def inference(self, x, reuse = False):
        """
        Defines the model used for inference. Output of this model is fed to the
        objective (or loss) function defined for the task.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        You can use the variable scope to activate/deactivate 'variable reuse'.

        Args:
           x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]
           reuse: Python bool to switch reusing on/off.

        Returns:
           l2_out: L2-normalized output tensor of shape [batch_size, 192]

        Hint: Parameter reuse indicates whether the inference graph should use
        parameter sharing or not. You can study how to implement parameter sharing
        in TensorFlow from the following sources:

        https://www.tensorflow.org/versions/r0.11/how_tos/variable_scope/index.html
        """
        with tf.variable_scope('ConvNet', reuse=reuse) as conv_scope:
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            with tf.name_scope('conv1') as scope:
                W_conv = tf.get_variable("w_conv1", [5, 5, 3, 64], initializer= initializers.xavier_initializer())
                b_conv = tf.get_variable("b_conv1", [64], initializer=tf.constant_initializer(0.0))
    
                conv = tf.nn.conv2d(x, W_conv, strides=[1, 1, 1, 1], padding='SAME')
                relu = tf.nn.relu(tf.nn.bias_add(conv, b_conv))
                max_pool = tf.nn.max_pool(relu , ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            with tf.name_scope('conv2') as scope:
                W_conv = tf.get_variable("w_conv2", [5, 5, 64, 64], initializer= initializers.xavier_initializer())
                b_conv = tf.get_variable("b_conv2", [64], initializer=tf.constant_initializer(0.0))
    
                conv = tf.nn.conv2d(max_pool, W_conv, strides=[1, 1, 1, 1], padding='SAME')
                relu = tf.nn.relu(tf.nn.bias_add(conv, b_conv))
                max_pool = tf.nn.max_pool(relu , ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
                
            with tf.name_scope('flatten') as scope:
                flat = tf.reshape(max_pool, [-1, 64*8*8], name='flatten')
     
            with tf.name_scope('fc1') as scope:
                W = tf.get_variable('w1',
                                    initializer=initializers.xavier_initializer(),
                                    shape=[flat.get_shape()[1].value, 384])
                tf.histogram_summary('weights1', W)
                b = tf.Variable(tf.zeros([384]))
                tf.histogram_summary('biasses1',  b)
                h = tf.nn.relu(tf.matmul(flat, W) + b, name='h1')
    
            with tf.name_scope('fc2') as scope:
                W = tf.get_variable('w2',
                                    initializer=initializers.xavier_initializer(),
                                    shape=[384, 192])
                tf.histogram_summary('weights2', W)
                b = tf.Variable(tf.zeros([192]))
                tf.histogram_summary('biasses2',  b)
                h = tf.nn.relu(tf.matmul(h, W) + b, name='h2')	
    
            with tf.name_scope('l2_norm') as scope:
                l2_out = tf.nn.l2_normalize(h, dim=0, name='l2_norm')
            
            
            ########################
            # END OF YOUR CODE    #
            ########################

        return l2_out

    def loss(self, channel_1, channel_2, label, margin):
        """
        Defines the contrastive loss. This loss ties the outputs of
        the branches to compute the following:

               L =  Y * d^2 + (1-Y) * max(margin - d^2, 0)

               where d is the L2 distance between the given
               input pair s.t. d = ||x_1 - x_2||_2 and Y is
               label associated with the pair of input tensors.
               Y is 1 if the inputs belong to the same class in
               CIFAR10 and is 0 otherwise.

               For more information please see:
               http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Args:
            channel_1: output of first channel (i.e. branch_1),
                              tensor of size [batch_size, 192]
            channel_2: output of second channel (i.e. branch_2),
                              tensor of size [batch_size, 192]
            label: Tensor of shape [batch_size]
            margin: Margin of the contrastive loss

        Returns:
            loss: scalar float Tensor
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        d = tf.reduce_sum(tf.square(tf.sub(channel_1, channel_2)), 1)
        d_square = tf.square(d)
        
        loss = label*d_square + tf.maximum(0., margin -d_square) * (1-label)
	
        loss = tf.reduce_mean(loss)
	
	#d = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(channel_1,channel_2),2),1,keep_dims=True))
	#tmp= label *tf.square(d)
    
    	#tmp2 = (1-label) *tf.square(tf.maximum((1 - d),0))
	#loss =  tf.reduce_mean(tmp +tmp2)
        
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
