from problems.adding_problem import *
from networks.keras_lstm import KerasLSTM

num_samples = 10000
seq_len = 50
adding_problem_dataset = AddingProblemDataset(num_samples, seq_len)

def test_keras_lstm():
	input_dim = 2
	output_dim = 1
	hidden_size = 128
	timesteps = seq_len
	batch_size = seq_len
	epochs = 20

	keras_lstm = KerasLSTM(input_dim, output_dim, hidden_size, timesteps)
	keras_lstm.train(adding_problem_dataset, batch_size, epochs)

test_keras_lstm()