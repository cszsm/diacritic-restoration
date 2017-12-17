from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.callbacks import EarlyStopping


class Network:
    def __init__(self, logger):

        self.logger = logger

        self.model = Sequential()
        self.model.add(LSTM(100, input_shape=(600, 30), return_sequences=True))
        self.model.add(LSTM(10, return_sequences=True))
        self.model.add(Dense(5))
        self.model.add(Activation('sigmoid'))

        self.model.compile(
            loss='binary_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

    def train(self, train_x, test_x, train_y, test_y):

        early_stopping = EarlyStopping(monitor='val_loss', patience=0)
        self.model.fit(
            train_x,
            train_y,
            batch_size=32,
            epochs=100,
            callbacks=[early_stopping],
            verbose=1,
            validation_split=0.4)
        score = self.model.evaluate(test_x, test_y, batch_size=32)
        print(score)

    def get_model(self):
        return self.model
