import numpy as np
import tensorflow as tf

class TFRNN:
    def __init__(
        self,
        num_in,
        num_hidden, 
        num_out,
        num_desired=1, # class
        single_output=True,
        state_type = tf.float32,
        rnn_cell=tf.contrib.rnn.BasicRNNCell,
        activation_hidden=tf.tanh,
        activation_out=tf.identity,
        optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001),
        loss_function=tf.squared_difference):

        self.loss_list = []

        init_state_C = np.sqrt(3 / (2 * num_hidden))
      
        # OVDE TREBA DA SE PROSLEDI MIDZI I INPUT SIZE input_size = num_in AKO JE MIDZIN
        self.cell = rnn_cell(num_units = num_hidden, activation = activation_hidden)   
        # [batch_size, max_time, num_in]   
        self.input_x = tf.placeholder(tf.float32, [None, None, num_in], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_desired] if single_output else [None, None, num_desired], name="input_y")
        
        init_state_dim = [None]
        for x in list(self.cell.state_size):
            init_state_dim.append(x)
        # print(init_state_dim)

        self.init_state = tf.placeholder(state_type, init_state_dim, name="init_state")
        self.state_type = state_type

        # set up parameters
        self.w_ho = tf.get_variable("w_ho", shape=[num_out, self.cell.output_size], 
                                            initializer=tf.contrib.layers.xavier_initializer()) # fixme
        self.b_o = tf.Variable(tf.zeros(num_out, 1),
           dtype=tf.float32, name="b_o")

        # [batch_size, max_time, self.cell.output_size]
        outputs_h, state = tf.nn.dynamic_rnn(self.cell, self.input_x, # INITIAL STATE!!!
          dtype = tf.float32)

        # IF ONLY LAST
        if single_output:
            outputs_h = outputs_h[:, -1, :] # batch_size x 1 x self.cell.output_size
            # za sve ove 
            outputs_h = tf.transpose(tf.reshape(outputs_h, (-1, self.cell.output_size))) # cellout x batch
            outputs_o = tf.transpose(activation_out(tf.matmul(self.w_ho, outputs_h) + self.b_o))
        else:
            #outputs_h = tf.transpose(outputs_h)
            outputs_o = tf.transpose(activation_out(tf.matmul(tf.expand_dims(self.w_ho, 0), outputs_h) + self.b_o))
        # batch_size x num_out

        # losses and train step
        self.total_loss = tf.reduce_mean(loss_function(outputs_o, self.input_y))
        self.train_step = optimizer.minimize(self.total_loss, name='Optimizer')

    def train(self, dataset, batch_size, epochs):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.loss_list = []
            num_batches = dataset.get_batch_count(batch_size)
            X_test, Y_test = dataset.get_test_data()

            for epoch_idx in range(epochs):

                print("New epoch", epoch_idx)

                for batch_idx in range(num_batches):
                    X_batch, Y_batch = dataset.get_batch(batch_idx, batch_size)
                    # ne mzoe 
                    # init_state = np.zeros((batch_size, self.cell.state_size), dtype=self.state_type)
                    init_state_dim = [batch_size]
                    for x in list(self.cell.state_size):
                        init_state_dim.append(x)
                    # print(init_state_dim)
                    # print(self.state_type)
                    # init_state = tf.zeros(init_state_dim, dtype=self.state_type)
                    init_state = np.zeros(tuple(init_state_dim))
                    _total_loss, _ = sess.run([self.total_loss, self.train_step],
                        feed_dict={
                        self.input_x: X_batch,
                        self.input_y: Y_batch,
                        self.init_state: init_state
                        })

                    self.loss_list.append(_total_loss)

                    if batch_idx%10 == 0:
                        print("Step",batch_idx, "Loss", _total_loss)
                        # mozda ovde da ga evaluiram
                        # end of epoch eval

                init_state_dim = [X_test.shape[0]]
                for x in list(self.cell.state_size):
                    init_state_dim.append(x)
                # print(init_state_dim)
                # print(self.state_type)
                # init_state = tf.zeros(init_state_dim, dtype=self.state_type)
                init_state = np.zeros(tuple(init_state_dim))
                _total_loss, _ = sess.run([self.total_loss, self.train_step],
                    feed_dict={
                        self.input_x: X_test, # TODO ne test
                        self.input_y: Y_test,
                        self.init_state: init_state
                        })
                print("End of epoch, loss on training set: ", np.mean(self.loss_list), "loss on test set: ", _total_loss)

    def get_loss_list(self):
        return self.loss_list





