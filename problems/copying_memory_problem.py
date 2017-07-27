import numpy as np
from .dataset import Dataset

class CopyingMemoryProblemDataset(Dataset):
    def generate(self, num_samples):

        assert(self.sample_len > 20) #must be

        X = np.zeros((num_samples, self.sample_len, 1))
        rand = np.random.randint(low = 1, high = 8, size = (num_samples, 10, 1))
        X[:, :10] = rand
        X[:, -11] = 9
        
        Y = np.zeros((num_samples, self.sample_len, 1))
        Y[:, -10:] = X[:, :10]

        # for i in range(num_samples):
        #     half = int(self.sample_len / 2)
        #     first_i = np.random.randint(half)
        #     second_i = np.random.randint(half) + half
        #     X_mask[i, (first_i, second_i)] = 1
        #     Y[i, 0] = np.sum(X_value[i, (first_i, second_i)])
        # X = np.concatenate((X_value, X_mask), 2)
        return X, Y
