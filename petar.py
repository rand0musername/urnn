import tensorflow as tf
from problems.adding_problem import AddingProblemDataset
from problems.copying_memory_problem import CopyingMemoryProblemDataset
# from networks.keras_lstm import KerasLSTM
from networks.tf_rnn import TFRNN
from networks.urnn_cell import URNNCell

loss_path = 'results/'
def serialize_loss(loss, name):
    file = open(loss_path + name, 'w')
    for l in loss:
        file.write("%s\n" % l)

class Main:
    def init_data(self):
        print('Generating data...')
        # init adding problem
        adp_samples = 100000 # TODO check this
        self.ap_batch_size = 50
        self.ap_epochs = 10

        self.apd_timesteps = [100, 200, 400, 750]
        self.apds = [AddingProblemDataset(adp_samples, timesteps) for timesteps in self.apd_timesteps]
        self.dummy_apd = AddingProblemDataset(100, 50);

        # init copying memory problem
        cmd_samples = 100000 # TODO fix
        self.cmp_batch_size = 50 # TODO ???
        self.cmp_epochs = 10
        self.cmd_timesteps = [120, 220, 320, 520]
        self.cmds = [CopyingMemoryProblemDataset(cmd_samples, timesteps) for timesteps in self.cmd_timesteps]
        self.dummy_cmd = CopyingMemoryProblemDataset(100, 50)

        print('Done.')

    def train_network(self, net, dataset, batch_size, epochs):
        print('Traing', net.name, '...')
        net.train(dataset, batch_size, epochs)
        serialize_loss(net.get_loss_list(), net.name + str(dataset.get_sample_len()))
        print('Network', net.name, 'done.')

    # def test_dummy_networks():
    #     print('Training dummy networks...')
    #     tf.reset_default_graph()

    #     print('Traing dummies on adding problem...')
    #     for ap_net in self.ap_networks:
    #         self.train_network(ap_net, self.dummy_apd)
    #     print('Dummy adding problem done.')

    #     print('Traing dummies on cpoying memory problem...')
    #     for cmp_net in self.cmp_networks:
    #         self.train_network(cmp_net, self.dummy_cmd)
    #     print('Dummy adding problem done.')
    #     print('Testing dummy networks done!')

    def train_networks_for_timestep_idx(self, idx):
        print('Initializing networks...')

        # self.ap_networks = []
        # self.cmp_networks = []

        tf.reset_default_graph()

        self.ap_simplernn = TFRNN(
            name="ap_simplernn",
            num_in = 2,
            num_hidden = 128,
            num_out = 1,
            num_target = 1,
            single_output = True,
            state_type = tf.float32,
            rnn_cell=tf.contrib.rnn.BasicRNNCell,
            activation_hidden=tf.tanh,
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
            loss_function=tf.squared_difference)
        #self.ap_networks.append(self.ap_simplernn)
        self.train_network(self.ap_simplernn, self.apds[idx], self.ap_batch_size, self.ap_epochs)

        tf.reset_default_graph()

        self.ap_lstm = TFRNN(
            name="ap_lstm",
            num_in = 2,
            num_hidden = 128,
            num_out = 1,
            num_target = 1,
            single_output = True,
            state_type = tf.float32,
            rnn_cell=tf.contrib.rnn.LSTMCell,
            activation_hidden=tf.tanh,
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
            loss_function=tf.squared_difference)
        #self.ap_networks.append(self.ap_lstm)
        self.train_network(self.ap_lstm, self.apds[idx], self.ap_batch_size, self.ap_epochs)

        tf.reset_default_graph()

        # TODO insert urnn here

        # ---------------------
        # self.ap_networks.append(self.ap_urnn)

        self.cmp_simple_rnn = TFRNN(
            name="cmp_simple_rnn",
            num_in = 1,
            num_hidden = 80,
            num_out = 10,
            num_target = 1,
            single_output = False,
            state_type = tf.float32,
            rnn_cell=tf.contrib.rnn.BasicRNNCell,
            activation_hidden=tf.tanh,
            activation_out=tf.identity,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
            loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits)
        #self.cmp_networks.append(self.cmp_simple_rnn)
        self.train_network(self.cmp_simple_rnn, self.cmds[idx], self.cmp_batch_size, self.cmp_epochs)

        tf.reset_default_graph()

        self.cmp_lstm = TFRNN(
            name="cmp_lstm",
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
        #self.cmp_networks.append(self.cmp_lstm)
        self.train_network(self.cmp_lstm, self.cmds[idx], self.cmp_batch_size, self.cmp_epochs)

        # TODO insert urnn here

        # ---------------------
        # self.cmp_networks.append(self.cmp_urnn)

        # self.cmp_urnn = TFRNN(
        #     name="cmp_urnn",
        #     num_in = 1,
        #     num_hidden = 128,
        #     num_out = 10,
        #     num_target = 1,
        #     single_output = False,
        #     state_type = tf.float32,
        #     rnn_cell=URNNCell,
        #     activation_hidden=None, # mod relu
        #     activation_out=tf.identity,
        #     optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
        #     loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits)

        print('Done.')

    # def train_networks(self, timesteps_idx):
    #     self.train_networks_adp(timesteps_idx)
    #     self.train_networks_cmp(timesteps_idx)

    # def train_networks_ap(self, idx):
    #     ap_batch_size = 50
    #     ap_epochs = 20
    #     print('Traing networks on adding problem with timelag =', self.apds[idx].get_sample_len())
    #     for net in self.ap_networks:
    #         self.train_network(net, self.apds[idx], ap_batch_size, ap_epochs)
    #     print('Finished training networks on adding problem with timelag =', self.apds[idx].get_sample_len())

    # def train_networks_cmp(self, idx):
    #     cmp_batch_size = 50
    #     cmp_epochs = 20
    #     print('Traing networks on adding problem with timelag =', self.cmds[idx].get_sample_len())
    #     for net in self.cmp_networks:
    #         self.train_network(net, self.cmds[idx], cmp_batch_size, cmp_epochs)
    #     print('Finished training networks on adding problem with timelag =', self.cmds[idx].get_sample_len())

    def train_networks(self):
        print('Staring training...')

        #self.test_dummy_networks()

        timesteps_idx = 4
        for i in range(timesteps_idx):
            main.train_networks_for_timestep_idx(i)

        print('Done and done.')

main = Main()
main.init_data()
main.train_networks()
# main.train_networks()
# main.test_cpm()
#main.test_dummy_networks()

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
