import numpy as np
"""
This module implements various losses for the network.
You should fill in code into indicated sections. 
"""

def HingeLoss(x, y):
  """
  Computes hinge loss and gradient of the loss with the respect to the input for multiclass SVM.
  Args:
    x: Input data.
    y: Labels of data. 
  Returns:
    loss: Scalar hinge loss.
    dx: Gradient of the loss with the respect to the input x.
  """
  ########################################################################################
  # TODO:                                                                                #
  # Compute hinge loss on input x and y and store it in loss variable. Compute gradient  #
  # of the loss with respect to the input and store it in dx variable.                   #
  ########################################################################################
  dx = None
  loss = None
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx

def CrossEntropyLoss(x, y):
  """
  Computes cross entropy loss and gradient with the respect to the input.
  Args:
    x: Input data.
    y: Labels of data. 
  Returns:
    loss: Scalar cross entropy loss.
    dx: Gradient of the loss with the respect to the input x.
  
  """
  ########################################################################################
  # TODO:                                                                                #
  # Compute cross entropy loss on input x and y and store it in loss variable. Compute   #
  # gradient of the loss with respect to the input and store it in dx.                   #
  ########################################################################################
  dx = None
  loss = None
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx


def SoftMaxLoss(x, y):
  """
  Computes the loss and gradient with the respect to the input for softmax classfier.
  Args:
    x: Input data.
    y: Labels of data. 
  Returns:
    loss: Scalar softmax loss.
    dx: Gradient of the loss with the respect to the input x.
  """
  ########################################################################################
  # TODO:                                                                                #
  # Compute softmax loss on input x and y and store it in loss variable. Compute gradient#
  # of the loss with respect to the input and store it in dx variable.                   #
  ########################################################################################
  # to calculate the softmax loss, we need a ground truth matrix with 1's for true classes
  # and 0's elsewhere, dimensions are batch_size x nr_classes
  m = np.zeros((y.shape[0], x.shape[1])) # initialize ground truth matrix
  # fill m with 1's at location given by labels in y
  m = np.array([[int(i == label) for i in range(x.shape[1])] for label in y])

  # softmax loss:
  # calculate p by taking exp(x)/sum(exp(x))
  p = np.divide(np.exp(x).T, np.sum(np.exp(x), axis=1)).T
  # gradient toward input: p_i - y_i, so we substract p by the ground truth matrix
  dx = p - m
  # loss: minus the sum of all log(p) values where ground truth is 1, averaged over
  # the mini-batch
  # we are only interested in the log(p) values where the ground truth is 1
  p_true = np.multiply(np.log(p), m)
  loss = -np.mean(np.sum(p_true, axis=1))
  ########################################################################################
  #                              END OF YOUR CODE                                        #
  ########################################################################################

  return loss, dx
