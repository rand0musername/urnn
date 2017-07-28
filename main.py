import tensorflow as tf
from problems.adding_problem import AddingProblemDataset
from problems.copying_memory_problem import CopyingMemoryProblemDataset
# from networks.keras_lstm import KerasLSTM
from networks.tf_rnn import TFRNN

class Main:

    def init_data(self):
        print('Generating data...')
        # init adding problem
        adp_samples = 10000 # check this
        self.adp_timesteps = [100, 200, 400, 750]
        self.apds = [AddingProblemDataset(adp_samples, timesteps) for timesteps in self.adp_timesteps]

        # init copying memory problem
        cmd_samples = 10000
        self.cmd_timesteps = [100, 200, 300, 500]
        self.cmds = [CopyingMemoryProblemDataset(cmd_samples, timesteps) for timesteps in self.cmd_timesteps]

        print('Done.')

    def init_networks(self):
        # TODO finish this function
        print('Initializing networks...')

        # self.ap_lstm = TFRNN(
        #     num_in = 2,
        #     num_hidden = 128,
        #     num_out = 1,
        #     num_target = 1,
        #     single_output = True,
        #     state_type = tf.float32,
        #     rnn_cell=tf.contrib.rnn.LSTMCell,
        #     activation_hidden=tf.tanh,
        #     activation_out=tf.identity,
        #     optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
        #     loss_function=tf.squared_difference)

        self.cmp_lstm = TFRNN(
            num_in = 1,
            num_hidden = 40,
            num_out = 10,
            num_target = 1,
            single_output = False,
            state_type = tf.float32,
            rnn_cell=tf.contrib.rnn.LSTMCell,
            activation_hidden=tf.tanh,
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
            loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits)

        print('Done.')

    def train_networks(self):
        print('Staring training...')

        batch_size = 20
        epochs = 20

        # loss = self.ap_lstm.get_loss_list()
        # self.ap_lstm.train(AddingProblemDataset(10000, 100), 100, 40)
        # loss = self.ap_lstm.get_loss_list()

        self.cmp_lstm.train(self.cmds[0], 50, 40)
        loss = self.cmp_lstm.get_loss_list()

        file = open('some_loss.txt', 'w')
        for item in loss:
            file.write("%s\n" % item)

        print('Done.')

main = Main()
main.init_data()
main.init_networks()
main.train_networks()


"""
def test_keras_lstm():
    input_dim = 2
    output_dim = 1
    hidden_size = 128
    timesteps = seq_len
    batch_size = seq_len
    epochs = 20

    rnn = AddingProblemURNN(input_dim, hiddAddingProblemURNNen_size, output_dim)
    rnn.train(adding_problem_dataset, batch_size, epochs)


    keras_lstm = KerasLSTM(input_dim, output_dim, hidden_size, timesteps)
    keras_lstm.train(adding_problem_dataset, batch_size, epochs)
"""
