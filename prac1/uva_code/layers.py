import numpy as np
"""
This module implements various layers for the network.
You should fill in code into indicated sections.
"""

class Layer(object):
  """
  Base class for all layers classes.
  """
  def __init__(self, layer_params = None):
    """
    Initializes the layer according to layer parameters.
    Args:
      layer_params: Dictionary with parameters for the layer.
    """
    self.train_mode = False

  def initialize(self):
    """
    Cleans cache. Cache stores intermediate variables needed for backward computation. 
    """
    self.cache = None

  def layer_loss(self):
    """
    Returns partial loss of layer parameters for regularization term of full loss.
    
    Returns:
      loss: Partial loss of layer parameters.
    """
    return 0.

  def set_train_mode(self):
    """
    Sets train mode for the layer.
    """
    self.train_mode = True

  def set_test_mode(self):
    """
    Sets test mode for the layer.
    """
    self.train_mode = False

  def forward(self, X):
    """
    Forward pass.
    Args:
      x: Input to the layer.
  
    Returns:
      out: Output of the layer.
    """
    raise NotImplementedError("Forward pass is not implemented for base Layer class.")

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: Gradients of the previous layer.
    
    Returns:
      dx: Gradient of the output with respect to the input of the layer.
    """
    raise NotImplementedError("Backward pass is not implemented for base Layer class.")

class LinearLayer(Layer):
  """
  Linear layer.
  """
  def __init__(self, layer_params):
    """
    Initializes the layer according to layer parameters.
    Args:
      layer_params: Dictionary with parameters for the layer:
          input_size - input dimension;
          output_size - output dimension;
          weight_decay - regularization parameter for the weights;
          weight_scale - scale of normal distrubtion to initialize weights.
      
    """
    self.layer_params = layer_params
    self.layer_params.setdefault('weight_decay', 0.0)
    self.layer_params.setdefault('weight_scale', 0.001)

    self.params = {'w': None, 'b': None}
    self.grads = {'w': None, 'b': None}

    self.train_mode = False

  def initialize(self):
    """
    Initializes weights and biases. Cleans cache.
    Cache stores intermediate variables needed for backward computation.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Initialize weights self.params['w'] using normal distribution with mean = 0 and      #
    # std = self.layer_params['weight_scale'].                                             #
    #                                                                                      #
    # Initialize biases self.params['b'] with 0.                                           #
    ######################################################################################## 
    # weights are samples from a random normal distribution with mean 0, std=weight_scale,
    # and with dimensions output x input
    self.params['w'] = np.random.normal(0, self.layer_params['weight_scale'],
                                        [self.layer_params['output_size'],
                                         self.layer_params['input_size']])
    # biases are initialized as a zero vector with size=ouput
    self.params['b'] = np.zeros(self.layer_params['output_size'])
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    
    self.cache = None

  def layer_loss(self):
    """
    Returns partial loss of layer parameters for regularization term of full loss.
    
    Returns:
      loss: Partial loss of layer parameters.
    """

    ########################################################################################
    # TODO:                                                                                #
    # Compute the loss of the layer which responsible for L2 regularization term. Store it #
    # in loss variable.                                                                    #
    ########################################################################################
    # L2 regularization: 1/2 * weight_decay * weight^2
    loss = float(1/2) * self.layer_params['weight_decay'] * np.sum(np.power
                                                                   (self.params['w'], 2))
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    return loss

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
    # Implement forward pass for LinearLayer. Store output of the layer in out variable.   #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ######################################################################################## 
    # output of a LinearLayer: input * weights + bias
    out = np.dot(x, self.params['w'].T) + self.params['b']
    # Cache if in train mode
    if self.train_mode:
      self.cache = x # store input for backward pass
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: Gradients of the previous layer.
    
    Returns:
      dx: Gradients with respect to the input of the layer.
    """
    if not self.train_mode:
      raise ValueError("Backward is not possible in test mode")

    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for LinearLayer. Store gradient of the loss with respect to  #
    # layer parameters in self.grads['w'] and self.grads['b']. Store gradient of the loss  #
    # with respect to the input in dx variable.                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ######################################################################################## 
    # gradient loss toward input: gradient previous layer * weights
    dx = np.dot(dout, self.params['w'])
    # weight gradient: dout*input + regularization using weight decay*weights
    # note: weight gradients are divided by the batch size to be able to sum up the
    # gradients during the backward pass
    w_grad = np.power(self.cache.shape[0], -1.) * np.dot(dout.T, self.cache)
    w_reg = self.layer_params['weight_decay']*self.params['w']
    self.grads['w'] = w_grad + w_reg
    # bias gradient: mean of delta_out per class
    self.grads['b'] = -np.mean(dout, axis=0)
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class ReLULayer(Layer):
  """
  ReLU activation layer
  """
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
    # Implement forward pass for ReLULayer. Store output of the layer in out variable.     #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ######################################################################################## 
    # output of ReLU Layer: theshold input at 0
    out = np.clip(x, 0, None) # threshold input with lower bound 0 and no upper bound
    # Cache if in train mode
    if self.train_mode:
      self.cache = out # store output for backward pass
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    
    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: Gradients of the previous layer.
    
    Returns:
      dx: Gradients with respect to the input of the layer.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for ReLULayer. Store gradient of the loss with respect to    #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ######################################################################################## 
    der_relu = np.clip(self.cache, 0, 1) # compute derivate of ReLU by clipping 0, 1
    dx = np.multiply(der_relu, dout) # multiply derivate ReLU with dout
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class SigmoidLayer(Layer):
  """
  Sigmoid activation layer.
  """
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
    # Implement forward pass for SigmoidLayer. Store output of the layer in out variable.  #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    # sigmoid: 1/1+exp(x)
    out = np.divide(1., 1. + np.exp(x)) #compute sigmoid activation

    # Cache if in train mode
    if self.train_mode:
      self.cache = out # store output of the layer
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    
    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: Gradients of the previous layer.
    
    Returns:
      dx: Gradients with respect to the input of the layer.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for SigmoidLayer. Store gradient of the loss with respect to #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################
    # derivative sigmoid: f'(x) = f(x)*(1-f(x))
    der_sig = np.multiply(self.cache, (1-self.cache)) # compute sigmoid derivative
    dx = np.multiply(der_sig, dout) # gradient is derivative times gradients previous layer
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class TanhLayer(Layer):
  """
  Tanh activation layer.
  """
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
    # Implement forward pass for TanhLayer. Store output of the layer in out variable.     #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    # tanh = 2/(1+exp(-2x))-1
    out = np.divide(2., 1 + np.exp(-2.*x))-1 # compute tanh activation

    # Cache if in train mode
    if self.train_mode:
      self.cache = out # store activation for backward pass
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    
    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: Gradients of the previous layer.
    
    Returns:
      dx: Gradients with respect to the input of the layer.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for TanhLayer. Store gradient of the loss with respect to    #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################
    # derivative tanh: 1 - tanh^2
    der_tanh = 1 - np.power(self.cache, 2) # compute tanh derivative
    dx = np.multiply(der_tanh, dout) # derivative times gradients
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class ELULayer(Layer):
  """
  ELU activation layer.
  """

  def __init__(self, layer_params):
    """
    Initializes the layer according to layer parameters.
    Args:
      layer_params: Dictionary with parameters for the layer:
          alpha - alpha parameter;
      
    """
    self.layer_params = layer_params
    self.layer_params.setdefault('alpha', 1.0)
    self.train_mode = False

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
    # Implement forward pass for ELULayer. Store output of the layer in out variable.      #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    # ELU: alpha(exp(x)-1) if x < 0, x if x => 0
    first_elu = np.multiply(self.layer_params['alpha']*(np.exp(x)-1.), x < 0)
    second_elu = np.multiply(x, x >= 0)
    out = first_elu + second_elu

    # Cache if in train mode
    if self.train_mode:
      self.cache = x
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################
    
    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: Gradients of the previous layer.
    
    Returns:
      dx: Gradients with respect to the input of the layer.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for ELULayer. Store gradient of the loss with respect to     #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################
    # derivative of ELU(x): ELU(x)+alpha if x < 0, 1 if x => 0
    alpha = self.layer_params['alpha']
    elu = alpha*(np.exp(self.cache)-1)
    first_der = np.multiply(elu+alpha, x < 0)
    second_der = self.cache >= 0
    elu_der = first_der + second_der
    dx = np.multiply(elu_der, dout)
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx

class SoftMaxLayer(Layer):
  """
  Softmax activation layer.
  """

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
    # Implement forward pass for SoftMaxLayer. Store output of the layer in out variable.  #
    #                                                                                      #
    # Hint: You can store intermediate variables in self.cache which can be used in        #
    # backward pass computation.                                                           #
    ########################################################################################
    out = None

    # Cache if in train mode
    if self.train_mode:
      self.cache = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return out
  
  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: Gradients of the previous layer.
    
    Returns:
      dx: Gradients with respect to the input of the layer.
    """
    ########################################################################################
    # TODO:                                                                                #
    # Implement backward pass for SoftMaxLayer. Store gradient of the loss with respect to #
    # the input in dx variable.                                                            #
    #                                                                                      #
    # Hint: Use self.cache from forward pass.                                              #
    ########################################################################################s
    dx = None
    ########################################################################################
    #                              END OF YOUR CODE                                        #
    ########################################################################################

    return dx
