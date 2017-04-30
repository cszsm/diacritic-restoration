import random

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

# import logger


class Network:

    vowel = ''

    def __init__(self, params, vowel, logger):

        self.vowel = vowel
        self.units = params['units']

        print(str(self.units) + 'units')
        self.logger = logger

        self.model = Sequential()
        self.model.add(LSTM(self.units, return_sequences=False, input_shape=(3, 30)))

        output_length = 2
        if self.vowel in ['o', 'u']:
            output_length = 4

        self.model.add(Dense(output_length, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    def run(self, train_x, test_x, train_y, test_y):

        self.logger.log('\nvowel: ' + self.vowel)

        early_stopping = EarlyStopping(monitor='loss', patience=0)
        self.model.fit(train_x, train_y, batch_size=32, epochs=100, callbacks=[early_stopping], verbose=3)
        score = self.model.evaluate(test_x, test_y, batch_size=32)

        self.logger.log('\nloss: ')
        self.logger.log(str(score[0]))
        self.logger.log('\naccuracy: ')
        self.logger.log(str(score[1]))


    def get_model(self):
        return self.model

    @staticmethod
    def get_random_parameters():
        params = {}
        params['units'] = random.randrange(10, 100, 10)
        return params

    @staticmethod
    def log_parameters(l, params):
        l.log('\n\nunits: ' + str(params['units']))
