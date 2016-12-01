from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



class ConvNet(object):
    """
   This class implements a convolutional neural network in TensorFlow.
   It incorporates a certain graph model to be trained and to be used
   in inference.
    """

    def __init__(self, n_classes = 10):
        """
        Constructor for an ConvNet object. Default values should be used as hints for
        the usage of each parameter.
        Args:
          n_classes: int, number of classes of the classification problem.
                          This number is required in order to specify the
                          output dimensions of the ConvNet.
        """
        self.n_classes = n_classes

    def inference(self, x):
        """
        Performs inference given an input tensor. This is the central portion
        of the network where we describe the computation graph. Here an input
        tensor undergoes a series of convolution, pooling and nonlinear operations
        as defined in this method. For the details of the model, please
        see assignment file.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        Although the model(s) which are within the scope of this class do not require
        parameter sharing it is a good practice to use variable scope to encapsulate
        model.

        Args:
          x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

        Returns:
          logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
                  the logits outputs (before softmax transformation) of the
                  network. These logits can then be used with loss and accuracy
                  to evaluate the model.
        """
        with tf.variable_scope('ConvNet'):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
        
            reg_strength = 0.001
            with tf.name_scope('conv1') as scope:
                W_conv = tf.get_variable("w_conv1", [5, 5, 3, 64], initializer= tf.random_normal_initializer(), regularizer=tf.contrib.layers.l2_regularizer(reg_strength))
                b_conv = tf.get_variable("b_conv1", [64], initializer=tf.constant_initializer(0.0))
    
                conv = tf.nn.conv2d(x, W_conv, strides=[1, 1, 1, 1], padding='SAME')
                relu = tf.nn.relu(tf.nn._bias_add(conv + b_conv))
                max_pool = tf.nn.max_pool(relu , ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    	
            with tf.name_scope('conv2') as scope:
                W_conv = tf.get_variable("w_conv2", [5, 5, 64, 64], initializer= tf.random_normal_initializer(), regularizer=tf.contrib.layers.l2_regularizer(reg_strength))
                b_conv = tf.get_variable("b_conv2", [64], initializer=tf.constant_initializer(0.0))
    
                conv = tf.nn.conv2d(max_pool, W_conv, strides=[1, 1, 1, 1], padding='SAME')
                relu = tf.nn.relu(tf.nn.bias_add(conv, b_conv))
                max_pool = tf.nn.max_pool(relu , ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    	
            with tf.name_scope('fc1') as scope:
                W = tf.get_variable('w1',
                                    initializer=tf.random_normal_initializer(),
                                    shape=[max_pool.get_shape()[1].value, 384],
    				regularizer=tf.contrib.layers.l2_regularizer(reg_strength))
                tf.histogram_summary('weights1', W)
                b = tf.Variable(tf.zeros([384]))
                tf.histogram_summary('biasses1',  b)
                h = tf.relu(tf.matmul(max_pool, W) + b, name=scope.name)
    
            with tf.name_scope('fc2') as scope:
                W = tf.get_variable('w2',
                                    initializer=tf.random_normal_initializer(),
                                    shape=[384, 192],
                                    regularizer=tf.contrib.layers.l2_regularizer(reg_strength))
                tf.histogram_summary('weights2', W)
                b = tf.Variable(tf.zeros([192]))
                tf.histogram_summary('biasses2',  b)
                h = tf.nn.relu(tf.matmul(h, W) + b, name=scope.name)	
    
            with tf.name_scope('fc3') as scope:
                W = tf.get_variable('final_w',
                                    initializer=tf.random_normal_initializer(),
                                    shape=[192, 10],
                                    regularizer=tf.contrib.layers.l2_regularizer(reg_strength))
                tf.histogram_summary('final_weights', W)
                b = tf.Variable(tf.zeros([10]))
                tf.histogram_summary('final_biasses',  b)
                logits = tf.matmul(h, W) + b
            ########################
            # END OF YOUR CODE    #
            ########################
        return logits

    def accuracy(self, logits, labels):
        """
        Calculate the prediction accuracy, i.e. the average correct predictions
        of the network.
        As in self.loss above, you can use tf.scalar_summary to save
        scalar summaries of accuracy for later use with the TensorBoard.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                     with one-hot encoding. Ground truth labels for
                     each observation in batch.

        Returns:
          accuracy: scalar float Tensor, the accuracy of predictions,
                    i.e. the average correct predictions over the whole batch.
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
	 	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	  	tf.scalar_summary('accuracy', accuracy)
        ########################
        # END OF YOUR CODE    #
        ########################

        return accuracy

    def loss(self, logits, labels):
        """
        Calculates the multiclass cross-entropy loss from the logits predictions and
        the ground truth labels. The function will also add the regularization
        loss from network weights to the total loss that is return.
        In order to implement this function you should have a look at
        tf.nn.softmax_cross_entropy_with_logits.
        You can use tf.scalar_summary to save scalar summaries of
        cross-entropy loss, regularization loss, and full loss (both summed)
        for use with TensorBoard. This will be useful for compiling your report.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                       with one-hot encoding. Ground truth labels for each
                       observation in batch.

        Returns:
          loss: scalar float Tensor, full loss = cross_entropy + reg_loss
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
    	with tf.name_scope('cross_entropy'):
        	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)) 
        	tf.scalar_summary('cross entropy', loss)
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss
