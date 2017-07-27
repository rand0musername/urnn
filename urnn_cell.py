import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
 
class DiagonalMatrix():
    def __init__(self, name, num_units):
        init_w = tf.random_uniform([num_units], minval=-np.pi, maxval=np.pi)
        self.w = tf.Variable(init_w, name=name)
        self.vec = tf.complex(tf.cos(self.w), tf.sin(self.w))

    # batch_sz x num_units -> transform every row
    def mul(self, z): # [num_units] * [batch_sz * num_units] -> [batch_sz * num_units]
        return vec_c * z

class ReflectionMatrix():
    def __init__(self, name, num_units):
        self.re = tf.Variable(tf.random_uniform([num_units], minval=-1, maxval=1), name=name+"_re")
        self.im = tf.Variable(tf.random_uniform([num_units], minval=-1, maxval=1), name=name+"_im")
        self.v = tf.complex(self.re, self.im)
        # NORMALIZE?!

    # batch_sz x num_units -> transform every row
    def mul(self, z): # [num_units] * [batch_sz * num_units] -> [batch_sz * num_units]
        vstar = tf.conj(v) # [num_units] 
        sq_norm = tf.reduce_sum(v)**2 #[1] 
        prod = v * tf.reduce_sum(tvstar * z, 1) #[num_units]
        return z - prod * (2 / sq_norm) #[num_units]

# FFTs are constant: 0 parameters

# does stuff over rows

def fft(mat):
    # batch_sz x num_units
    # scale???
    # transpose???
    return tf.fft(mat)/sqrt(mat.shape[0])

def ifft(mat):
    #  batch_sz x num_units
    # scale???
    # transpose???
    return tf.ifft(mat)/sqrt(mat.shape[0])


###################################################################################################333

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
        self._num_units = num_units
        self._input_size = input_size

        self._state_size = num_units 
        self._output_size = num_units*2

        self.activation = activation # DON'T PASS THIS
        # ENTER THE LAYER: 2*num_units, real and complex
        self.w_ih = tf.get_variable("w_ih", shape=[2*num_units, input_size], 
                                    initializer=tf.contrib.layers.xavier_initializer()) # fixme

        self.b_h = tf.Variable(tf.zeros(num_units), # state size actually
                                    name="b_h")
        # num_units
        self.D1 = DiagonalMatrix("D1", num_units)
        self.R1 = ReflectionMatrix("R1", num_units)

        tf.set_random_seed(int(datetime.now().timestamp()))

        self.D2 = DiagonalMatrix("D2", num_units)
        self.R2 = ReflectionMatrix("R2", num_units)
        self.D3 = DiagonalMatrix("D3", num_units)

    @property
    def input_size(self):
        return self._input_size # real

    @property
    def state_size(self):
        return self._state_size # complex

    @property
    def output_size(self):
        return self._output_size # real

    # z: complex: batch_sz x num_units
    # bias: num_units
    def modReLU(self, z, bias):
        EPS = 1e-6 # hack?
        norm = tf.complex_abs(z)
        if (norm + bias) >= 0:
            scale = (norm+bias) / (n+EPS)
            return f.complex(tf.real(z)*scale, tf.imag(z)*scale)
        else:
            return 0

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

        # INPUT
        inputs_mul = tf.matmul(inputs, tf.transpose(self.w_ih)) # batch_sz x 2*num_units
        inputs_mul_c = tf.complex( inputs_mul[:, :self.state_size], 
                                   inputs_mul[:, self.state_size:] )
        # batch_sz x num_units

        # STATE: all complex
        # batch_sz x num_units !!!!!!!!!!!!!!!!
        state_mul = self.D1.mul(state)

        state_mul = fft(state_mul)

        state_mul = self.R1.mul(state_mul)

        state_mul = tf.transpose(tf.random_shuffle(tf.transpose(state_mul))) # permuts columns

        state_mul = self.D2.mul(state_mul)

        state_mul = ifft(state_mul)

        state_mul = self.R2.mul(state_mul)


        state_mul = self.D3.mul(state_mul) 
        # batch_sz x num_units !!!!!!!!!!!!!!!!
        
        # FINALIZE
        preact = inputs_mul_c + state_mul
        # batch_sz x num_units

        out_state = modReLU(preact, self.b_h) # bsz x numunits complex
        output = tf.concat([tf.real(out_state), tf.imag(out_state)], 1)

        # output is 2*num_units R, but the outside network is ready for that (uses self.output_size everywhere)
            # maybe this is wrong but nvm for now
        # out_state is num_units C
        return output, out_state
