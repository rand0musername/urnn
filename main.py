from problems.adding_problem import *
#from networks.keras_lstm import KerasLSTM
from tf_rnn import AddingProblemRNN
from tf_lstm import AddingProblemLSTM
from tf_urnn import AddingProblemURNN

num_samples = 10000
seq_len = 20
adding_problem_dataset = AddingProblemDataset(num_samples, seq_len)

def test_tf_lstm():
    input_dim = 2
    output_dim = 1
    hidden_size = 128
    batch_size = 20
    epochs = 20
    rnn = AddingProblemRNN(input_dim, hidden_size, output_dim)
    rnn.train(adding_problem_dataset, batch_size, epochs)

"""
def test_keras_lstm():
	input_dim = 2
	output_dim = 1
	hidden_size = 128
	timesteps = seq_len
	batch_size = seq_len
	epochs = 20

	keras_lstm = KerasLSTM(input_dim, output_dim, hidden_size, timesteps)
	keras_lstm.train(adding_problem_dataset, batch_size, epochs)
"""
test_tf_lstm()
#test_keras_lstm()