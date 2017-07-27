# from .tf_rnn import TFRNN

# class Networks():
#     def __init__(self):
#         tf_lstm = TFRNN(
#                  num_in,
#                  num_hidden, 
#                  num_out,
#                  num_desired=1, # class
#                  single_output=True,
#                  rnn_cell=tf.contrib.rnn.BasicRNNCell,
#                  activation_hidden=tf.tanh,
#                  activation_out=tf.identity,
#                  optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001))

#     def train(self):
#         raise NotImplementedError()