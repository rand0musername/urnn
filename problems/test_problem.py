import numpy as np
#from .dataset import Dataset


class DataProvider():
    def __init__(self, examples, sequence_length, inputs, outputs):
        self.create(examples, sequence_length, inputs, outputs)


    def get_example_count(self):
        return self._x.shape[0]
    def create(self, examples, sequence_length, inputs, outputs):
        np.random.seed(1) # Set the seed explicitly so that the same data is generated every time.
        self._x = np.random.randn(examples, sequence_length, inputs)  # random values of input length
        self._z = np.zeros((examples, sequence_length, outputs))
        self._z[:, 2::3, 0] = 1  # learn to count to three regardless of input x
        
        np.random.seed(2)
        self._x_test = np.random.randn(examples, sequence_length, inputs)  # random values of input length
        self._z_test = np.zeros((examples, sequence_length, outputs))
        self._z_test[:, 2::3, 0] = 1  # learn to count to three regardless of input x

    def load_data(self):
        return self._x, self._z, self._x_test, self._z_test
