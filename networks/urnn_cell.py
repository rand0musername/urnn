import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
 
# Diagonal unitary matrix
class DiagonalMatrix():
    def __init__(self, name, num_units):
        init_w = tf.random_uniform([num_units], minval=-np.pi, maxval=np.pi)
        self.w = tf.Variable(init_w, name=name)
        self.vec = tf.complex(tf.cos(self.w), tf.sin(self.w))

    # [batch_sz, num_units]
    def mul(self, z): 
        # [num_units] * [batch_sz, num_units] -> [batch_sz, num_units]
        return self.vec * z

# Reflection unitary matrix
class ReflectionMatrix():
    def __init__(self, name, num_units):
        self.re = tf.Variable(tf.random_uniform([num_units], minval=-1, maxval=1), name=name+"_re")
        self.im = tf.Variable(tf.random_uniform([num_units], minval=-1, maxval=1), name=name+"_im")
        self.v = tf.complex(self.re, self.im) # [num_units]
        self.vstar = tf.conj(self.v) # [num_units]
        # normalize?!

    # [batch_sz, num_units]
    def mul(self, z): 
        # [num_units] * [batch_sz * num_units] -> [batch_sz * num_units]
        sq_norm_ind = tf.complex_abs(self.v)**2
        sq_norm = tf.reduce_sum(sq_norm_ind, 1) # [batch_sz]
        
        vstar_z = tf.reduce_sum(self.vstar * z, 1) # [batch_sz]
        prod = self.v * tf.tile(vstar_z, [1, num_units]) # [batch_sz * num_units]
        return z - 2 * prod / sq_norm # [batch_sz * num_units]

# FFTs
# z: complex[batch_sz, num_units]
# does FFT over rows, transpose?!
# no scaling?!

def FFT(z):
    return tf.fft(z) / sqrt(z.shape[1])

def IFTT(z):
    return tf.ifft(z) / sqrt(z.shape[1])

# z: complex[batch_sz, num_units]
# bias: real[num_units]

def modReLU(self, z, bias):
    EPS = 1e-6 # hack?
    norm = tf.complex_abs(z)
    if (norm + bias) >= 0:
        scale = (norm + bias) / (norm + EPS)
        return tf.complex(tf.real(z)*scale, tf.imag(z)*scale)
    else:
        return tf.zeros(tf.shape(z))

###################################################################################################333

class URNNCell(tf.contrib.rnn.RNNCell):
    """The most basic URNN cell.
    Args:
        num_units (int): The number of units in the LSTM cell, hidden layer size.
        num_in: Input vector size, input layer size.
    """

    def __init__(self, num_units, num_in, reuse=None):
        super(URNNCell, self).__init__(_reuse=reuse)
        
        # save class variables
        self.num_in = num_in
        self._num_units = num_units
        self._state_size = num_units 
        self._output_size = num_units*2

        # set up input -> hidden connection
        self.w_ih = tf.get_variable("w_ih", shape=[2*num_units, input_size], 
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.b_h = tf.Variable(tf.zeros(num_units), # state size actually
                                    name="b_h")

        # elementary unitary matrices to get the big one
        self.D1 = DiagonalMatrix("D1", num_units)
        self.R1 = ReflectionMatrix("R1", num_units)
        self.D2 = DiagonalMatrix("D2", num_units)
        self.R2 = ReflectionMatrix("R2", num_units)
        self.D3 = DiagonalMatrix("D3", num_units)
        self.P = np.random.permutation(num_units)

    # needed properties

    @property
    def input_size(self):
        return self._input_size # real

    @property
    def state_size(self):
        return self._state_size # complex

    @property
    def output_size(self):
        return self._output_size # real

    def call(self, inputs, state):
        """The most basic URNN cell.
        Args:
            inputs (Tensor - batch_sz x input_size): One batch of cell input.
            state (Tensor - batch_sz x num_units): Previous cell state: COMPLEX
        Returns:
        A tuple (outputs, state):
            outputs (Tensor - batch_sz x num_units*2): Cell outputs on the whole batch.
            state (Tensor - batch_sz x num_units): New state of the cell.
        """

        # prepare input linear combination
        inputs_mul = tf.matmul(inputs, tf.transpose(self.w_ih)) # [batch_sz, 2*num_units]
        inputs_mul_c = tf.complex( inputs_mul[:, :self.num_units], 
                                   inputs_mul[:, self.num_units:] ) 
        # [batch_sz, num_units]
        
        # prepare state linear combination (always complex!)
        # [batch_sz, num_units]
        state_mul = self.D1.mul(state)
        state_mul = FFT(state_mul)
        state_mul = self.R1.mul(state_mul)
        state_mul = state_mul[:, P] # permutation
        state_mul = self.D2.mul(state_mul)
        state_mul = IFFT(state_mul)
        state_mul = self.R2.mul(state_mul)
        state_mul = self.D3.mul(state_mul) 
        # [batch_sz, num_units]
        
        # calculate preactivation
        preact = inputs_mul_c + state_mul
        # [batch_sz, num_units]

        new_state = modReLU(preact, self.b_h) # [batch_sz, num_units] C
        output = tf.concat([tf.real(new_state), tf.imag(new_state)], 1) # [batch_sz, 2*num_units] R
        # outside network (last dense layer) is ready for 2*num_units -> num_out
        
        return output, new_state
