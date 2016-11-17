import numpy as np
"""
This module implements Network model of the network.
You should fill in code into indicated sections.
"""

class Network(object):

  """
  Implements model of the network.
  """
  def __init__(self):
    """
    Initializes the layer according to layer parameters.
    """
    self.layers = []

  def add_layer(self, layer):
    """
    Adds layer to the network.
    Args:
      layer: Layer to put into the network.
    """
    self.layers.append(layer)

  def add_loss(self, loss):
    """
    Adds loss layer to the network.
    Args:
      loss: Loss to put into the network.
    """
    self.loss_func = loss

  def reset(self):
    """
    Resets network by reinitializing every layer.
    """
    for layer in self.layers:
      layer.initialize()

  def set_train_mode(self):
    """
    Sets train mode for the model.
    """
    for layer in self.layers:
      layer.set_train_mode()

  def set_test_mode(self):
    """
    Sets test mode for the model.
    """
    for layer in self.layers:
      layer.set_test_mode()

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: Input to the layer.
    Returns:
      out: Output of the layer.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement forward pass for the network. Store output of the network in out variable. #
    ########################################################################################
    # propagate input x to the next layer
    for layer in self.layers:
      x = layer.forward(x)
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return x

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: Gradients of the loss.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for the network.                                             #
    ########################################################################################
    # backwards propagation of delta dout to the previous layer
    for layer in reversed(self.layers):
      dout = layer.backward(dout)
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return

  def loss(self, out, y):
    """
    Computes loss and gradient of the loss with the respect to the input data.
    Args:
      out: Output of the network after forward pass.
      y: Labels of data.
    Returns:
      loss: Scalar loss.
      dout: Gradient of the loss with the respect to the input x.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Compute loss and gradient of the loss with the respect to output. Store them in loss #
    # and dout variables respectively.                                                     #
    ########################################################################################
    # get loss and gradient from output layer
    output_loss, output_gradient = self.loss_func(out, y)
    temp_loss = output_loss
    for layer in self.layers:
      temp_loss += layer.layer_loss() # add layer loss per layer
    loss = temp_loss
    dout = output_gradient
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return loss, dout
