from problems.adding_problem import *
from lstm.lstm import LSTM

num_samples = 100000
input_dim = 2
# output_dim = 250
timesteps = 20
batch_size = 20
epochs = 10

apd = AddingProblemDataset()
apd.create(num_samples, timesteps)

l = LSTM(batch_size, epochs, timesteps, input_dim)
l.run(apd)

# Testing LSTM on fake problem

# outputs = 1
# inputs = 5
# examples = 1000
# sequence_length = 10
# from problems.test_problem import DataProvider
# dp = DataProvider(examples, sequence_length, 1, outputs)
# l2 = LSTM(examples, 100, sequence_length, 1)
# l2.run(dp)
