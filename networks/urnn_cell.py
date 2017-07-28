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
        self.num_units = num_units

        self.re = tf.Variable(tf.random_uniform([num_units], minval=-1, maxval=1), name=name+"_re")
        self.im = tf.Variable(tf.random_uniform([num_units], minval=-1, maxval=1), name=name+"_im")
        self.v = tf.complex(self.re, self.im) # [num_units]
        self.vstar = tf.conj(self.v) # [num_units]
        # normalize?!

    # [batch_sz, num_units]
    def mul(self, z): 
        # [num_units] * [batch_sz * num_units] -> [batch_sz * num_units]
        sq_norm_ind = tf.abs(self.v)**2
        sq_norm = tf.reduce_sum(sq_norm_ind) # [1]
        
        vstar_z = tf.reduce_sum(self.vstar * z, 1) # [batch_sz]
        vstar_z = tf.reshape(vstar_z, [-1, 1]) # [batch_sz * 1]

        prod = self.v * tf.tile(vstar_z, [1, self.num_units]) # [batch_sz * num_units]
        
        return z - 2 * prod / tf.complex(sq_norm, 0.0) # [batch_sz * num_units]

# Permutation unitary matrix
class PermutationMatrix:
    def __init__(self, name, num_units):
        self.num_units = num_units
        perm = np.random.permutation(num_units)
        i_perm = np.arange(num_units)
        self.P = tf.constant(perm, tf.int32)
        self.E = tf.constant(i_perm, tf.int32)
        self.i = [[elem] for elem in range(num_units)]

    # [batch_sz, num_units], permute columns
    def mul(self, z): 
        # return tf.transpose(tf.gather(tf.transpose(z), self.P))
        z = tf.transpose(z)
        parts = tf.dynamic_partition(z, self.P, self.num_units)
        stitched = tf.dynamic_stitch(self.i, parts)
        stitched = tf.reshape(stitched, [self.num_units, -1])
        return tf.transpose(stitched)

# FFTs
# z: complex[batch_sz, num_units]
# does FFT over rows, transpose?!
# no scaling?!

def FFT(z):
    return tf.fft(z) / tf.sqrt(tf.complex(tf.cast(z.shape[1], tf.float32), 0.0))

def IFFT(z):
    return tf.ifft(z) / tf.sqrt(tf.complex(tf.cast(z.shape[1], tf.float32), 0.0))

# z: complex[batch_sz, num_units]
# bias: real[num_units]
def modReLU(z, bias): # relu(|z|+b) * (z / |z|)
    norm = tf.abs(z)
    scale = tf.nn.relu(norm + bias) / (norm + 1e-6)
    scaled = tf.complex(tf.real(z)*scale, tf.imag(z)*scale)
    return scaled

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
        self._num_in = num_in
        self._num_units = num_units
        self._state_size = num_units 
        self._output_size = num_units*2

        # set up input -> hidden connection
        self.w_ih = tf.get_variable("w_ih", shape=[2*num_units, num_in], 
                                    initializer=tf.contrib.layers.xavier_initializer())
        self.b_h = tf.Variable(tf.zeros(num_units), # state size actually
                                    name="b_h")

        # elementary unitary matrices to get the big one
        self.D1 = DiagonalMatrix("D1", num_units)
        self.R1 = ReflectionMatrix("R1", num_units)
        self.D2 = DiagonalMatrix("D2", num_units)
        self.R2 = ReflectionMatrix("R2", num_units)
        self.D3 = DiagonalMatrix("D3", num_units)
        self.P = PermutationMatrix("P", num_units)
    # needed properties

    @property
    def input_size(self):
        return self._num_in # real

    @property
    def state_size(self):
        return self._state_size # complex

    @property
    def output_size(self):
        return self._output_size # real

    def call(self, inputs, state):
        """The most basic URNN cell.
        Args:
            inputs (Tensor - batch_sz x num_in): One batch of cell input.
            state (Tensor - batch_sz x num_units): Previous cell state: COMPLEX
        Returns:
        A tuple (outputs, state):
            outputs (Tensor - batch_sz x num_units*2): Cell outputs on the whole batch.
            state (Tensor - batch_sz x num_units): New state of the cell.
        """
        print("cell.call inputs:", inputs.shape, inputs.dtype)
        print("cell.call state:", state.shape, state.dtype)

        # prepare input linear combination
        inputs_mul = tf.matmul(inputs, tf.transpose(self.w_ih)) # [batch_sz, 2*num_units]
        inputs_mul_c = tf.complex( inputs_mul[:, :self._num_units], 
                                   inputs_mul[:, self._num_units:] ) 
        # [batch_sz, num_units]
        
        # prepare state linear combination (always complex!)
        # [batch_sz, num_units]
        state_mul = self.D1.mul(state)
        state_mul = FFT(state_mul)
        state_mul = self.R1.mul(state_mul)
        state_mul = self.P.mul(state_mul)
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

        print("cell.call output:", output.shape, output.dtype)
        print("cell.call new_state:", new_state.shape, new_state.dtype)

        return output, new_state
