import keras

class LSTM():

    def __init__(self, batch_size, epochs, timestamps, input_dim):
        self.hidden_size = 128
        self.batch_size = batch_size
        self.epochs = epochs

        self.timestamps = timestamps
        self.input_dim = input_dim

        self.input = keras.layers.Input(
            (self.timestamps, self.input_dim))

        self.hidden = keras.layers.recurrent.LSTM(
            self.hidden_size,
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences = False)(self.input)

        self.output = keras.layers.Dense(
            1, 
            activation = None)(self.hidden)

        self.model = keras.models.Model(self.input, self.output)

        # inp = keras.layers.Input(shape=(batch_size, 2, 1))
        # flat = keras.layers.Flatten()(inp)
        # hidden1 = keras.layers.recurrent.LSTM(128)(flat)
        # self.model = Model(inputs = inp, outputs = hidden1)

    def run(self, dataset):
        X_train, Y_train, X_test, Y_test = dataset.load_data()

        print(X_train.shape)
        print(X_train)

        self.model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(lr = 0.001))
        self.model.fit(X_train, Y_test, batch_size = self.batch_size, epochs = self.epochs,
            validation_data = (X_test, Y_test))

        score = self.model.evaluate(X_test, Y_test, verbose = 1)

        print("Accuracy:", score)