from problems.adding_problem import AddingProblemDataset
from problems.copying_memory_problem import CopyingMemoryProblemDataset
# #from networks.keras_lstm import KerasLSTM
# from tf_rnn import AddingProblemRNN
# from tf_lstm import AddingProblemLSTM
# from tf_urnn import AddingProblemURNN
import tensorflow as tf
from networks.tf_rnn import TFRNN

class Main:
    def init_data(self):
        print('Generating data...')
        # init adding problem
        adp_samples = 10000 #check this
        self.adp_timesteps = [100, 200, 400, 750]
        self.apds = [AddingProblemDataset(adp_samples, timesteps) for timesteps in self.adp_timesteps]

        # init copying memory problem
        cmd_samples = 10000
        self.cmd_timesteps = [100, 200, 300, 500]
        self.cmds = [CopyingMemoryProblemDataset(cmd_samples, timesteps) for timesteps in self.cmd_timesteps]

        print('Done.')

    def init_networks(self): # FINISH THIS SHIT IS NOT DONE
        print('Initializing networks...')
        batch_size = 20
        epochs = 20
        #rnn = AddingProblemRNN(input_dim, hidden_size, output_dim)
        ap_lstm = TFRNN(
            num_in = 2,
            num_hidden = 128,
            num_out = 1,
            num_desired = 1,
            single_output = True,
            rnn_cell=tf.contrib.rnn.LSTMCell,
            activation_hidden=tf.tanh,
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
            loss_function=tf.squared_difference)

        ap_basic_rnn = TFRNN(
            num_in = 2,
            num_hidden = 128,
            num_out = 1,
            num_desired = 1,
            single_output = True,
            rnn_cell=tf.contrib.rnn.BasicRNNCell,
            activation_hidden=tf.tanh,
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
            loss_function=tf.squared_difference)


        # ap_lstm.train(adding_problem_dataset, batch_size, epochs)
        # ap_basic_rnn.train(adding_problem_dataset, batch_size, epochs)
        # ap_basic_rnn_loss = ap_basic_rnn.get_loss_list()

        # file = open('ap_basic_rnn_loss.txt', 'w')
        # for item in ap_basic_rnn_loss:
        #     file.write("%s\n" % item)


        # ap_lstm.train(adding_problem_dataset, batch_size, epochs)
        # ap_lstm_loss = ap_lstm.get_loss_list()

        # file = open('ap_ap_lstm_loss.txt', 'w')
        # for item in ap_lstm_loss:
        #     file.write("%s\n" % item)
        print('Done.')

main = Main()
main.init_data()
main.init_networks()


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