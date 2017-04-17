import random

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

import logger


class Network:

    def __init__(self, logger):

        self.units = random.randrange(10, 100, 10)

        print(str(self.units) + 'units')
        # self.logger = logger.Logger('/logs/' + str(self.units) + 'units')
        self.logger = logger

        self.model = Sequential()
        self.model.add(LSTM(self.units, return_sequences=False, input_shape=(3, 30)))
        self.model.add(Dense(2, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    def run(self, train_x, test_x, train_y, test_y):

        print('units: ' + str(self.units))
        self.logger.log('\nunits: ' + str(self.units))

        early_stopping = EarlyStopping(monitor='loss', patience=0)
        self.model.fit(train_x, train_y, batch_size=32, epochs=100, callbacks=[early_stopping], verbose=3)
        score = self.model.evaluate(test_x, test_y, batch_size=32)

        print('\nscore:')
        print(self.model.metrics_names)
        print(score)
        self.logger.log('\nloss: ')
        self.logger.log(str(score[0]))
        self.logger.log('\naccuracy: ')
        self.logger.log(str(score[1]))
        # for line in score:
        #     self.logger.log(line)


    def get_model(self):
        return self.model
