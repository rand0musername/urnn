import numpy as np
#from .dataset import Dataset

def serialize(X, path):
	f = open(path, 'w')
	print(X.shape, file = f)
	print(np.array_str(X), file = f)

def deserialize(path):
	f = open(path, 'r')
	shape = tuple(f.readline())
	nums = list(f.readline())
	return np.array(nums).resize(tuple(shape))

class AddingProblemDataset():
	def __init__(self):
		self.path_train_x = 'problems/adding_problem_data/X_train.ogi'
		self.path_train_y = 'problems/adding_problem_data/Y_train.ogi'
		self.path_test_x = 'problems/adding_problem_data/X_test.ogi'
		self.path_test_y = 'problems/adding_problem_data/Y_test.ogi'

	def generate(self, N, high = 1):
	    X_value = np.random.uniform(low = 0, high = high, size = (N, self.length, 1))
	    X_mask = np.zeros((N, self.length, 1))
	    Y = np.ones((N, 1))
	    for i in range(N):
	        positions = np.random.choice(self.length, size = 2, replace = False)
	        X_mask[i, positions] = 1
	        Y[i, 0] = np.sum(X_value[i, positions])
	    X = np.append(X_value, X_mask, axis = 2)
	    return X, Y

	def generate_training(self, N):
		X, Y = self.generate(N)
		serialize(X, self.path_train_x)
		serialize(Y, self.path_train_y)


	def generate_test(self, N):
		X, Y = self.generate(N)
		serialize(X, self.path_test_x)
		serialize(Y, self.path_test_y)



	def create(self, N, length = 30):
		self.N = N
		self.length = length
		pass
		# self.generate_training(N)
		# self.generate_test(N)

	def load_data(self):
		# X_train = deserialize(self.path_train_x)
		# Y_train = deserialize(self.path_train_y)
		# X_test = deserialize(self.path_test_x)
		# Y_test = deserialize(self.path_test_y)
		X_train, Y_train = self.generate(self.N)
		X_test, Y_test = self.generate(int(self.N))

		return X_train, Y_train, X_test, Y_test
# def test(N):
# 	a = AddingProblemDataset()
# 	a.create(1)
# 	x, y, x1, y1 = a.load_data()
# 	print(x, y)

# test(1)
