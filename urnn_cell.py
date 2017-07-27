import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class URNNCell(tf.contrib.rnn.RNNCell):
  """The most basic RNN cell.
  Args:
    num_units: int, The number of units in the LSTM cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
  """

  def __init__(self, num_units, activation=None, reuse=None):
    super(URNNCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    dummy = tf.matmul(inputs, tf.ones([inputs.shape[1], self._num_units]))
    output = tf.ones_like(dummy)
    return output, output
