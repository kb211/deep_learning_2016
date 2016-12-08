from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import cifar10_utils
import tensorflow as tf
import numpy as np
import convnet
from sklearn import manifold
import matplotlib
matplotlib.use('Agg')
import pylab as plt
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

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
    Performs training and evaluation of ConvNet model.

    First define your graph using class ConvNet and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    x_test, y_test = cifar10.test.images, cifar10.test.labels

    model = convnet.ConvNet(n_classes=10)
  
    x = tf.placeholder(tf.float32, [None, 32, 32, 3]) 
    y = tf.placeholder(tf.float32, [None, 10])
  
    logits = model.inference(x)  
  
    loss = model.loss(logits, y)
  
    accuracy = model.accuracy(logits, y)
  
    step = train_step(loss)
    
    init = tf.initialize_all_variables()
  
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
  
    merged = tf.merge_all_summaries()
  
    #train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train',
    #                                  sess.graph)
    #test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test')
    
    for i in range(FLAGS.max_steps):
      batch_xs, batch_ys = cifar10.train.next_batch(FLAGS.batch_size)
      summary, _ = sess.run([merged, step], feed_dict={x: batch_xs, y: batch_ys})
      #train_writer.add_summary(summary, i)
      
      if i % 100 == 0:
          summary, acc, l = sess.run([merged, accuracy, loss], feed_dict={x: x_test, y: y_test})
          print('iteration: ' + str(i) + 'Accuracy: ' + str(acc) + 'Loss: ' + str(l))
          #test_writer.add_summary(summary, i)

    #test_writer.close()
    #train_writer.close()
    
    save_path = saver.save(sess, "checkpoints/convnet")
    ########################
    # END OF YOUR CODE    #
    ########################


def train_siamese():
    """
    Performs training and evaluation of Siamese model.

    First define your graph using class Siamese and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    On train set, it is fine to monitor loss over minibatches. On the other
    hand, in order to evaluate on test set you will need to create a fixed
    validation set using the data sampling function you implement for siamese
    architecture. What you need to do is to iterate over all minibatches in
    the validation set and calculate the average loss over all minibatches.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    ########################


def feature_extraction():
    """
    This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
    model with restored parameters. From then on you can basically use that model in
    any way you want, for instance, feature extraction, finetuning or as a submodule
    of a larger architecture. However, this method should extract features from a
    specified layer and store them in data files such as '.h5', '.npy'/'.npz'
    depending on your preference. You will use those files later in the assignment.

    Args:
        [optional]
    Returns:
        None
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    model = convnet.ConvNet(n_classes=10)

    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, 10])

    logits = model.inference(x)

    accuracy = model.accuracy(logits, y)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()

    with tf.Session() as sess:
  	#saver.restore(sess, "checkpoints/convnet")

    	cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    	x_test, y_test = cifar10.test.images, cifar10.test.labels

    	#acc, fc2_out, fc1_out, flatten = sess.run([accuracy, model.fc2_out, model.fc1_out, model.flatten], feed_dict={x: x_test, y: y_test})
    	#print('Accuracy: ' + str(acc))
    
    
    	#tsne = manifold.TSNE(n_components=2 , init='pca', random_state=0)
    	#fc2_tsne = tsne.fit_transform(np.squeeze(fc2_out))
	#fc1_tsne = tsne.fit_transform(np.squeeze(fc1_out))
	#flatten_tsne = tsne.fit_transform(np.squeeze(flatten))
    	
	fc2_tsne = np.load('fc2_tsne')
	fc1_tsne = np.load('fc1_tsne')
	flatten_tsne = np.load('flatten_tsne')
	labels = np.argmax(y_test, axis=1)
	
    	plt.figure(figsize=(20, 20))  #in inches

        x = fc2_tsne[:,0]/np.linalg.norm(fc2_tsne[:,0])
	y = fc2_tsne[:,1]/np.linalg.norm(fc2_tsne[:,1])
        plt.scatter(x, y, c=labels)
    	plt.savefig('fc2_tsne_norm.png')

	plt.figure(figsize=(20, 20))

        x = fc1_tsne[:,0]/np.linalg.norm(fc1_tsne[:,0])
        y = fc1_tsne[:,1]/np.linalg.norm(fc1_tsne[:,1])
        plt.scatter(x, y, c=labels)
        plt.savefig('fc3_tsne_norm.png')
	
	plt.figure(figsize=(20, 20))  #in inches

        x = flatten_tsne[:,0]/np.linalg.norm(flatten_tsne[:,0])
        y = flatten_tsne[:,1]/np.linalg.norm(flatten_tsne[:,1])
        plt.scatter(x, y, c=labels)
        plt.savefig('flatten_tsne_norm.png')

	#fc2_tsne.dump('fc2_tsne')
	#fc1_tsne.dump('fc1_tsne')
	#flatten_tsne.dump('flatten_tsne')
	_classify(fc2_tsne, labels)
	_classify(fc3_tsne, labels)
	_classify(flatten_tsne, labels)
    ########################
    # END OF YOUR CODE    #
    ########################

def _classify(tsne, labels):
    from sklearn.svm import SVC
    for i in np.arange(0,10):
        classifier = SVC(kernel='linear', class_weight='balanced')  
        Y = [1 if label == i else 0 for label in labels  ]
    
        classifier.fit(tsne, Y)
    
        print('for class: ', i, 'score: ', classifier.score(tsne, labels))  

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

    if FLAGS.is_train == 'True':
        if FLAGS.train_model == 'linear':
            train()
        elif FLAGS.train_model == 'siamese':
            train_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")
    else:
        feature_extraction()

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
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')
    parser.add_argument('--is_train', type = str, default = True,
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
