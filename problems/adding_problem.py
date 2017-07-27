import numpy as np
from .dataset import Dataset

class AddingProblemDataset(Dataset):
    def __init__(self, num_samples, sample_len):
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.X_train, self.Y_train = self.generate(num_samples)
        self.X_test, self.Y_test = self.generate(int(num_samples / 5))

    def generate(self, num_samples):
        high = 1
        X_value = np.random.uniform(low = 0, high = high, size = (num_samples, self.sample_len, 1))
        X_mask = np.zeros((num_samples, self.sample_len, 1))
        Y = np.ones((num_samples, 1))
        for i in range(num_samples):
            half = int(self.sample_len / 2)
            first_i = np.random.randint(half)
            second_i = np.random.randint(half) + half
            X_mask[i, (first_i, second_i)] = 1
            Y[i, 0] = np.sum(X_value[i, (first_i, second_i)])
        X = np.concatenate((X_value, X_mask), 2)
        return X, Y

    def load_data(self):
        return self.X_train, self.Y_train, self.X_test, self.Y_test