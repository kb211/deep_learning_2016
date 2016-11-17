"""
This module implements training and evaluation of a multi-layer perceptron.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers
import numpy as np
import cifar10_utils
import mlp

# The default parameters are the same parameters that you used during practical 1.
# With these parameters you should get similar results as in the Numpy exercise.
### --- BEGIN default constants ---
LEARNING_RATE_DEFAULT = 2e-3
WEIGHT_REGULARIZER_STRENGTH_DEFAULT = 0.
WEIGHT_INITIALIZATION_SCALE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 200
MAX_STEPS_DEFAULT = 1500
DROPOUT_RATE_DEFAULT = 0.
DNN_HIDDEN_UNITS_DEFAULT = '100'
WEIGHT_INITIALIZATION_DEFAULT = 'normal'
WEIGHT_REGULARIZER_DEFAULT = 'l2'
ACTIVATION_DEFAULT = 'relu'
OPTIMIZER_DEFAULT = 'sgd'
### --- END default constants---

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs/cifar10'


# This is the list of options for command line arguments specified below using argparse.
# Make sure that all these options are available so we can automatically test your code
# through command line arguments.

# You can check the TensorFlow API at
# https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.layers.html#initializers
# https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#sharing-variables
WEIGHT_INITIALIZATION_DICT = {'xavier': lambda scale: initializers.xavier_initializer(), # Xavier initialisation
                              'normal': lambda scale: tf.random_normal_initializer(stddev=scale), # Initialization from a standard normal
                              'uniform': lambda scale: tf.random_uniform_initializer(minval=-scale, maxval=scale) # Initialization from a uniform distribution
                             }

# You can check the TensorFlow API at
# https://www.tensorflow.org/versions/r0.11/api_docs/python/contrib.layers.html#regularizers
# https://www.tensorflow.org/versions/r0.11/api_docs/python/state_ops.html#sharing-variables
WEIGHT_REGULARIZER_DICT = {'none': lambda weight: regularizers.l1_regularizer(0.), # No regularization
                           'l1': lambda weight: regularizers.l1_regularizer(weight), # L1 regularization
                           'l2': lambda weight: regularizers.l2_regularizer(weight) # L2 regularization
                          }

# You can check the TensorFlow API at
# https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#activation-functions
ACTIVATION_DICT = {'relu': tf.nn.relu, # ReLU
                   'elu': tf.nn.elu, # ELU
                   'tanh': tf.tanh, #Tanh
                   'sigmoid': tf.sigmoid} #Sigmoid

# You can check the TensorFlow API at
# https://www.tensorflow.org/versions/r0.11/api_docs/python/train.html#optimizers
OPTIMIZER_DICT = {'sgd': tf.train.GradientDescentOptimizer, # Gradient Descent
                  'adadelta': tf.train.AdadeltaOptimizer, # Adadelta
                  'adagrad': tf.train.AdagradOptimizer, # Adagrad
                  'adam': tf.train.AdamOptimizer, # Adam
                  'rmsprop': tf.train.RMSPropOptimizer # RMSprop
                  }

FLAGS = None

def train():
  """
  Performs training and evaluation of MLP model. Evaluate your model each 100 iterations
  as you did in the practical 1. This time evaluate your model on the whole test set.
  """
  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  tf.set_random_seed(42)
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
  x_test, y_test = cifar10.test.images, cifar10.test.labels
  
  _initializer  = FLAGS.weight_init
  _regularizer = FLAGS.weight_reg
  _activation = FLAGS.activation
  _optimizer = FLAGS.optimizer
#  model = mlp.MLP()
  model = mlp.MLP(
               n_classes=10,
               is_training=True,
               n_hidden = dnn_hidden_units,
               activation_fn = ACTIVATION_DICT[_activation],
               dropout_rate=FLAGS.dropout_rate, 
               weight_initializer = WEIGHT_INITIALIZATION_DICT[_initializer](FLAGS.weight_init_scale),
               weight_regularizer = WEIGHT_REGULARIZER_DICT[_regularizer](FLAGS.weight_reg_strength))
  
  x = tf.placeholder(tf.float32, [None, 3072]) 
  y = tf.placeholder(tf.float32, [None, 10])
  
  logits = model.inference(x)  
  
  loss = model.loss(logits, y)
  
  accuracy = model.accuracy(logits, y)
  
  #train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE_DEFAULT).minimize(cross_entropy)
  train_step = OPTIMIZER_DICT[_optimizer](FLAGS.learning_rate, name="optimizer").minimize(loss)
  init = tf.initialize_all_variables()
  
  sess = tf.Session()
  sess.run(init)
  
  merged = tf.merge_all_summaries()
  
  train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train',
                                      sess.graph)
  test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
  
  for i in range(MAX_STEPS_DEFAULT):
      batch_xs, batch_ys = cifar10.train.next_batch(BATCH_SIZE_DEFAULT)
      summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})
      train_writer.add_summary(summary, i)
      
      if i % 100 == 0:
          summary, acc = sess.run([merged, accuracy], feed_dict={x: x_test, y: y_test})
          print(acc)
          test_writer.add_summary(summary, i)

  test_writer.close()
  train_writer.close()
          
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main(_):
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  # Make directories if they do not exists yet
  if not tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.MakeDirs(FLAGS.log_dir)
  if not tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.MakeDirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--weight_init', type = str, default = WEIGHT_INITIALIZATION_DEFAULT,
                      help='Weight initialization type [xavier, normal, uniform].')
  parser.add_argument('--weight_init_scale', type = float, default = WEIGHT_INITIALIZATION_SCALE_DEFAULT,
                      help='Weight initialization scale (e.g. std of a Gaussian).')
  parser.add_argument('--weight_reg', type = str, default = WEIGHT_REGULARIZER_DEFAULT,
                      help='Regularizer type for weights of fully-connected layers [none, l1, l2].')
  parser.add_argument('--weight_reg_strength', type = float, default = WEIGHT_REGULARIZER_STRENGTH_DEFAULT,
                      help='Regularizer strength for weights of fully-connected layers.')
  parser.add_argument('--dropout_rate', type = float, default = DROPOUT_RATE_DEFAULT,
                      help='Dropout rate.')
  parser.add_argument('--activation', type = str, default = ACTIVATION_DEFAULT,
                      help='Activation function [relu, elu, tanh, sigmoid].')
  parser.add_argument('--optimizer', type = str, default = OPTIMIZER_DEFAULT,
                      help='Optimizer to use [sgd, adadelta, adagrad, adam, rmsprop].')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run()
