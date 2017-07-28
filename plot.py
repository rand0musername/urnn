import matplotlib.pyplot as plt

basic_rnn_loss = [float(l) for l in open('ap_basic_rnn_loss.txt', 'r').read().split('\n')]
lstm_loss = [float(l) for l in open('ap_ap_lstm_loss.txt', 'r').read().split('\n')]
lstm100_loss = [float(l) for l in open('lstm100_ap_loss.txt', 'r').read().split('\n')]

plt.plot(lstm100_loss, 'r-')
plt.plot(lstm_loss, 'b-')
plt.show()