import numpy as np
import copy
"""
This module implements Solver that optimize model parameters using provided optimizer.
You should fill in code into indicated sections. 
"""

class Solver(object):
  """
  Implements solver.
  """
  def __init__(self, model):
    """
    Initializes solver with the model.
    Args:
      model: Model to optimize.
    """
    self.model = model

  def _reset(self, optimizer, optimizer_config = {}):
    """
    Resets solver by reinitializing every layer in the model.
    Resets optimizer configuration for every layer in the model.
    """
    self.model.reset()

    self.optimizer = optimizer
    self.optimizer_configs = {}
    for i in range(len(self.model.layers)):
      if hasattr(self.model.layers[i], 'params'):
        self.optimizer_configs[i] = {}
        param_names = self.model.layers[i].params.keys()
        for param_name in param_names:
          self.optimizer_configs[i][param_name] = copy.deepcopy(optimizer_config)

  def train_on_batch(self, x_batch, y_batch):
    """
    Trains on batch.
    Args:
      x_batch: Input batch data.
      y_batch: Input batch labels.
    Returns:
      loss: Loss of the model.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Compute gradient of the loss on the batch with the respect to model parameters.      #
    # Compute gradient of the loss with respect ot parametrs of the model.                 #
    ########################################################################################
    # propogate input x_batch through the network
    out = self.model.forward(x_batch) # output is forward propagation
    # calculate loss and gradient of that loss
    loss, dout = self.model.loss(out, y_batch) # compute loss of output given batch labels
    self.model.backward(dout) # backwards propogation of gradient
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    for i in range(len(self.model.layers)):
      if hasattr(self.model.layers[i], 'params'):
        param_names = self.model.layers[i].params.keys()
        optimizer_config = self.optimizer_configs[i]
        for param_name in param_names:
          w = self.model.layers[i].params[param_name]
          dw = self.model.layers[i].grads[param_name]
          optimizer_config = self.optimizer_configs[i][param_name]
          next_w, next_optimizer_config = self.optimizer(w, dw, optimizer_config)
          self.model.layers[i].params[param_name] = next_w
          self.optimizer_configs[i][param_name] = next_optimizer_config

    return out, loss

  def test_on_batch(self, x_batch, y_batch):
    """
    Tests on batch.
    Args:
      x_batch: Input batch data.
      y_batch: Input batch labels.
    Returns:
      out: Output of the network for the provided batch.
      loss: Loss of the network for the provided batch.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Compute output and loss for x_batch and y_batch.                                     #
    ########################################################################################
    out = self.model.forward(x_batch) # output is forward propagation
    loss, _ = self.model.loss(out, y_batch) # calculate loss of output given batch labels
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out, loss

  def fit(self, x_train, y_train, optimizer, optimizer_config = {}, x_val = None, y_val = None,
          batch_size = 200, num_iterations = 1000, val_iteration = 100, verbose = False):
    """
    Fits model on x_train, y_train data using specified optimizer. If x_val and y_val are
    provided then also test on this data every val_iteration iterations.
    Args:
      x_train: Input train data.
      y_train: Input train labels.
      optimizer: Optimizer to use for optimizing model.
      optimizer_config: Configuration of optimizer.
      x_val: Input validation data.
      y_val: Input validation labels.
      batch_size: Batch size for training.
      num_iterations: Maximum number of iterations to perform.
      val_iteration: Perform validation every val_iteration iterations.
      verbose: Output or not intermediate results during training.
    Returns:
      train_loss_history: Train loss history during training of the model.
      train_acc_history: Train accuracy history during training of the model.
      val_loss_history: Validation loss history during training of the model.
      val_acc_history: Validation accuracy history during training of the model.
    """
    self._reset(optimizer, optimizer_config)

    train_acc_history = []
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    for iteration in xrange(num_iterations):

      ########################################################################################
      # TODO:                                                                                #
      # Sample a random mini-batch with size of batch_size from train set. Put images to     #
      # x_train_batch and labels to y_train_batch.                                           #
      ########################################################################################
      # get random indices to sample mini-batch
      # (mini-batch can not contain images more than once)
      idx = np.random.choice(x_train.shape[0], size=batch_size, replace=False)
      x_train_batch = x_train[idx]
      y_train_batch = y_train[idx]
      ########################################################################################
      #                              END OF YOUR CODE                                        #
      ########################################################################################

      self.model.set_train_mode()
      ########################################################################################
      # TODO:                                                                                #
      # Train on batch (x_train_batch, y_train_batch) using train_on_batch method. Compute   #
      # train loss and accuracy on this batch.                                               #
      ########################################################################################
      # train batch to get train output and loss of model
      t_out, t_loss = self.train_on_batch(x_train_batch, y_train_batch)
      train_loss = t_loss # loss from training on model
      # calculate accuracy using output and labels
      train_acc = self.accuracy(t_out, y_train_batch)
      ########################################################################################
      #                              END OF YOUR CODE                                        #
      ########################################################################################
      self.model.set_test_mode()

      if iteration % val_iteration  == 0 or iteration == num_iterations - 1:
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        if verbose:
          print("Iteration {0:d}/{1:d}: Train Loss = {2:.3f}, Train Accuracy = {3:.3f}".format(
               iteration, num_iterations, train_loss_history[-1], train_acc_history[-1]))

        if not x_val is None:
          ########################################################################################
          # TODO:                                                                                #
          # Compute the loss and accuracy on the validation set.                                 #
          ########################################################################################
          # test model on validation set
          v_out, v_loss = self.test_on_batch(x_val, y_val) # calculate output and loss
          val_loss = v_loss # set loss
          val_acc = self.accuracy(v_out, y_val) # compute accuracy on validation set
          ########################################################################################
          #                              END OF YOUR CODE                                        #
          ########################################################################################
          val_loss_history.append(val_loss)
          val_acc_history.append(val_acc)
          if verbose:
            print("Iteration {0:d}/{1:d}. Validation Loss = {2:.3f}, Validation Accuracy = {3:.3f}".format(
                 iteration, num_iterations, val_loss_history[-1], val_acc_history[-1]))

    if not x_val is None:
      return train_loss_history, train_acc_history, val_loss_history, val_acc_history
    else:
      return train_loss_history, train_acc_history

  def accuracy(self, out, y):
    """
    Computes accuracy on output out and y.
    Args:
      out: Output of the network.
      y: True labels.
    Returns:
      accuracy: Accuracy score.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Compute the accuracy on output of the network. Store it in accuracy variable.        #
    ########################################################################################
    # models predicts correct class when highest score in output coincides with label y
    scores = [np.argmax(out, axis=1) == y]
    accuracy = np.mean(scores) # accuracy is mean of scores
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    return accuracy

  def predict(self, x):
    """
    Computes predictions on x.
    Args:
      x: Input data.
    Returns:
      y_pred: Predictions on x.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Compute the prediction on data x. Store it in y_pred variable.                       #
    #                                                                                      #
    ########################################################################################
    # prediction on data x is found using the argmax of output of the forward pass
    out = self.model.forward(x) # get output from forward pass
    y_pred = np.argmax(out, axis=1) # get highest value for each data point
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return y_pred

  def score(self, x, y):
    """
    Computes accuracy score on x and y.
    Args:
      x: Input data.
      y: Input labels.
    Returns:
      score: Accuracy score.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Compute the accuracy on data x with labels y. Store it in score variable.            #
    ########################################################################################
    # score is accuracy of forward pass given label y
    score = self.accuracy(self.model.forward(x), y)
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return score
