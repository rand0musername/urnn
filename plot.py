import matplotlib.pyplot as plt

def deserialize(path):
	loss = [float(l) for l in open(path, 'r').read().split('\n')[:-1]]

ap_lstm100 = deserialize('presentation/')

plt.plot(lstm100_loss, 'r-')
plt.plot(lstm_loss, 'b-')
plt.show()