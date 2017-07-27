import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class AddingProblemLSTM:

    def __init__(self,
                 num_in,
                 num_hidden, 
                 num_out,
                 num_desired=1, # class
                 activation_hidden=tf.tanh,
                 activation_out=tf.identity,
                 optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001)): # pass dataset

        self.cell = tf.contrib.rnn.LSTMCell(num_units = num_hidden, activation = activation_hidden)   
        # [batch_size, max_time, num_in]   
        self.input_x = tf.placeholder(tf.float32, [None, None, num_in], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_desired], name="input_y")

        # set up parameters
        self.w_ho = tf.Variable(tf.random_uniform([num_out, self.cell.output_size], minval=-1, maxval=1), 
                                dtype=tf.float32, name="w_ho")
        self.b_o = tf.Variable(tf.zeros(num_out, 1),
                               dtype=tf.float32, name="b_o")

        # [batch_size, max_time, self.cell.output_size]
        outputs_h, state = tf.nn.dynamic_rnn(self.cell, self.input_x, # INITIAL STATE!!!
                                          dtype = tf.float32)

        outputs_h = outputs_h[:, -1, :] # batch_size x 1 x self.cell.output_size
        outputs_h = tf.transpose(tf.reshape(outputs_h, (-1, self.cell.output_size))) # cellout x batch
        outputs_o = tf.transpose(activation_out(tf.matmul(self.w_ho, outputs_h) + self.b_o))
        # batch_size x num_out

        # losses and train step
        self.total_loss = tf.reduce_mean(tf.squared_difference(outputs_o, self.input_y))
        self.train_step = optimizer.minimize(self.total_loss, name='Optimizer')

    def train(self, dataset, batch_size, num_epochs):
        # AddingProblemDataset(10000, 100)
        
        X_train, Y_train, X_test, Y_test = dataset.load_data()
        # 10k x time len x num_in
        # 10k x num_out

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_list = []
            num_batches = X_train.shape[0] // batch_size

            for epoch_idx in range(num_epochs):

                print("New epoch", epoch_idx)

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size

                    batchX = X_train[start_idx:end_idx, :, :]
                    batchY = Y_train[start_idx:end_idx, :]

                    _total_loss, _ = sess.run([self.total_loss, self.train_step],
                        feed_dict={
                            self.input_x: batchX,
                            self.input_y: batchY
                        })

                    loss_list.append(_total_loss)

                    if batch_idx%100 == 0:
                        print("Step",batch_idx, "Loss", _total_loss)
                # end of epoch eval
                _total_loss, _ = sess.run([self.total_loss, self.train_step],
                    feed_dict={
                        self.input_x: X_test,
                        self.input_y: Y_test
                    })
                print("End of epoch, loss on training set: ", np.mean(loss_list), "loss on test set: ", _total_loss)