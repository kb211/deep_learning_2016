from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import vgg
import tensorflow as tf
import numpy as np
import convnet
import cifar10_utils

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
REFINE_AFTER_K_STEPS_DEFAULT = 0

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

def train_step(loss):
    """
    Defines the ops to conduct an optimization step. You can set a learning
    rate scheduler or pick your favorite optimizer here. This set of operations
    should be applicable to both ConvNet() and Siamese() objects.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate, name="optimizer").minimize(loss)
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

def train():
    """
    Performs training and evaluation of your model.

    First define your graph using vgg.py with your fully connected layer.
    Then define necessary operations such as trainer (train_step in this case),
    savers and summarizers. Finally, initialize your model within a
    tf.Session and do the training.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every PRINT_FREQ iterations
    - on test set every EVAL_FREQ iterations

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    std = 0.001
    reg_strength = 0.0001

    x = tf.placeholder(tf.float32, [None, None, None, 3])

    y = tf.placeholder(tf.float32, [None, 10])
    
    model = convnet.ConvNet()
    pool5, assign_ops = vgg.load_pretrained_VGG16_pool5(x)

    #pool5 = tf.stop_gradient(pool5)

    with tf.variable_scope('fc') as var_scope:
        with tf.name_scope('flatten') as scope:
            flat = tf.reshape(pool5, [-1, pool5.get_shape()[3].value], name='flatten')

        with tf.name_scope('fc1') as scope:
            W = tf.get_variable('w1',
                initializer=tf.random_normal_initializer(stddev=std),
                shape=[flat.get_shape()[1].value, 384],
                regularizer=tf.contrib.layers.l2_regularizer(reg_strength))
                
            b = tf.Variable(tf.zeros([384]))
            h = tf.nn.relu(tf.matmul(flat, W) + b, name='h1')

        with tf.name_scope('fc2') as scope:
            W = tf.get_variable('w2',

	    initializer=tf.random_normal_initializer(stddev=std),
	    shape=[384, 192],
	    regularizer=tf.contrib.layers.l2_regularizer(reg_strength))
            b = tf.Variable(tf.zeros([192]))

            h2 = tf.nn.relu(tf.matmul(h, W) + b, name='h2')

        with tf.name_scope('fc3') as scope:
            W = tf.get_variable('final_w',

	    initializer=tf.random_normal_initializer(stddev=std),
	    shape=[192, 10],
	    regularizer=tf.contrib.layers.l2_regularizer(reg_strength))

            b = tf.Variable(tf.zeros([10]))

            logits = tf.matmul(h2, W) + b

    loss = model.loss(logits, y)
    accuracy = model.accuracy(logits, y)
    step = train_step(loss)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    #for op in assign_ops:
    sess.run(assign_ops)

    merged = tf.merge_all_summaries()

    train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/VGGtrain',
                                      sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/VGGtest')

    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    x_test, y_test = cifar10.test.images, cifar10.test.labels

    for i in range(FLAGS.max_steps):
        batch_xs, batch_ys = cifar10.train.next_batch(FLAGS.batch_size)
        summary, _ = sess.run([merged, step], feed_dict={x: batch_xs, y: batch_ys})

        if i % FLAGS.print_freq == 0 :
            train_writer.add_summary(summary, i)

        if i % FLAGS.eval_freq == 0:
            summary, acc, l = sess.run([merged, accuracy, loss], feed_dict={x: x_test[0:1000], y: y_test[0:1000]})

            print('iteration: ' + str(i) + 'Accuracy: ' + str(acc) + 'Loss: ' + str(l))
            
            test_writer.add_summary(summary, i)

    test_writer.close()
    train_writer.close()

          
    ########################
    # END OF YOUR CODE    #
    ########################

def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    print_flags()

    initialize_folders()
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')
    parser.add_argument('--refine_after_k', type = int, default = REFINE_AFTER_K_STEPS_DEFAULT,
                      help='Number of steps after which to refine VGG model parameters (default 0).')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
