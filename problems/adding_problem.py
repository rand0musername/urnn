import numpy as np
from .dataset import Dataset

class AddingProblemDataset(Dataset):
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
