class Dataset:
    def __init__(self, num_samples, sample_len):
        self.num_samples = num_samples
        self.sample_len = sample_len
        self.X_train, self.Y_train = self.generate(int(num_samples * 0.6))
        self.X_valid, self.Y_valid = self.generate(int(num_samples * 0.2))
        self.X_test, self.Y_test = self.generate(int(num_samples * 0.2))

    def generate(self, num_samples):
    	raise NotImplementedError()

    def get_data(self):
        return self.X_train, self.Y_train, self.X_valid, self.Y_valid, self.X_test, self.Y_test
    
    def get_validation_data(self):
        return self.X_valid, self.Y_valid

    def get_test_data(self):
        return self.X_test, self.Y_test
    
    def get_batch_count(self, batch_size):
        return self.X_train.shape[0] // batch_size

    def get_batch(self, batch_idx, batch_size):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        X_batch = self.X_train[start_idx:end_idx, :, :]
        Y_batch = self.Y_train[start_idx:end_idx, :]

        return X_batch, Y_batch