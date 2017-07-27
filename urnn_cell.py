import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
 
 # Matrix classes

class SimpleUnitaryMatrix:
    # this is a num_units X num_units Matrix represented by one vector
    def __init_(self, name, num_units):
        self.vec = tf.Variable(tf.zeros(num_units), name=name, dtype=tf.complex64)

    # multiply this Matrix with a [num_units X batch_size] Matrix
    # Matrix (batch_size row vectors) to return a Matrix with same shape
    def mul_batch(self, b):
        # b is a complex matrix, return a complex too
        raise NotImplementedError("To be implemented")

class DiagonalMatrix(SimpleUnitaryMatrix):
    def mul_batch(self, b):
        print('D')

class ReflectionMatrix(SimpleUnitaryMatrix):
    def mul_batch(self, b):
        print('R')

class PermutationMatrix(SimpleUnitaryMatrix):
    def mul_batch(self, b):
        print('P')

class FFTMatrix(SimpleUnitaryMatrix):
    def mul_batch(self, b):
        print('F')

class IFFTMatrix(SimpleUnitaryMatrix):
    def mul_batch(self, b):
        print('I')

class URNNCell(tf.contrib.rnn.RNNCell):
    """The most basic URNN cell.
    Args:
    num_units (int): The number of units in the LSTM cell, hidden layer size.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    """

    def __init__(self, num_units, input_size, activation=None, reuse=None):
        super(URNNCell, self).__init__(_reuse=reuse)
        self.num_units = num_units
        self.input_size = input_size
        self.activation = activation or math_ops.tanh
        # ENTER THE LAYER: 2*num_units, real and complex
        self.w_ih = tf.Variable(tf.random_uniform([2*num_units, input_size], minval=-1, maxval=1), 
                                    name="w_ho")
        self.b_h = tf.Variable(tf.zeros(num_units, 1),
                                    name="b_o")
        self.D1 = DiagonalMatrix("D1", num_units)
        self.F = FFTMatrix("F", num_units)
        self.R1 = ReflectionMatrix("R1", num_units)
        self.P = PermutationMatrix("P", num_units)
        self.D2 = DiagonalMatrix("D2", num_units)
        self.IF = IFFTMatrix("IF", num_units)
        self.R2 = ReflectionMatrix("R2", num_units)
        self.D3 = DiagonalMatrix("D3", num_units)

    @property
    def input_size(self):
        return self.input_size

    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units*2

    def modReLU(self, a):
        return 'x'

    def call(self, inputs, state):
        """The most basic URNN cell.
        Args:
        inputs (Tensor - batch_sz x input_size): One batch of cell input.
        state (Tensor - batch_sz x num_units): Previous cell state: COMPLEX
        Returns:
        A tuple (outputs, state):
        outputs (Tensor - batch_sz x num_units): Cell outputs on the whole batch.
        state (Tensor - batch_sz x num_units): New state of the cell.
        """

        # STATE
        state = tf.transpose(state) # num_units x batch_sz
        state_mul = self.D1.mul_batch(state)
        state_mul = self.F.mul_batch(state_mul)
        state_mul = self.R1.mul_batch(state_mul)
        state_mul = self.P.mul_batch(state_mul)
        state_mul = self.D2.mul_batch(state_mul)
        state_mul = self.IF.mul_batch(state_mul)
        state_mul = self.R2.mul_batch(state_mul)
        state_mul = self.D3.mul_batch(state_mul) # num_units x batch_sz

        # INPUT
        inputs = tf.transpose(inputs) # input_size x batch_sz
        inputs_mul = tf.matmul(self.w_ih, inputs) # 2*num_units x batch_sz
        inputs_mul = 
        in_proj_c = tf.complex( in_proj[:, :self.state_size], in_proj[:, self.state_size:] )
            
        # FINALIZE
        preact = state_mul + inputs_mul + self.b_h
        # num_units x batch_sz

        output = tf.transpose(self.activation(preact))
        return output, output

