import random

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

import keras.backend as K

from math import pow

class Network:

    vowel = ''

    def __init__(self, params, vowel, logger):

        self.vowel = vowel
        self.units = params['units']
        self.window_size = params['window_size']

        # print(str(self.units) + 'units')
        self.logger = logger

        self.model = Sequential()
        self.model.add(LSTM(self.units, return_sequences=False, input_shape=(self.window_size * 2, 30)))

        output_length = 2
        if self.vowel in ['o', 'u']:
            output_length = 4

        self.model.add(Dense(output_length))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy', precision, recall, f1_score])

    def run(self, train_x, test_x, train_y, test_y):

        self.logger.log('\nvowel: ' + self.vowel)

        early_stopping = EarlyStopping(monitor='loss', patience=0)
        self.model.fit(train_x, train_y, batch_size=32, epochs=100, callbacks=[early_stopping], verbose=3)
        score = self.model.evaluate(test_x, test_y, batch_size=32)

        # self.logger.log(self.model.metrics_names)
        for name in self.model.metrics_names:
            self.logger.log(name)
        self.logger.log('\nloss: ')
        self.logger.log(str(score[0]))
        self.logger.log('\naccuracy: ')
        self.logger.log(str(score[1]))
        # self.logger.log('\nfmeasure: ')
        # self.logger.log(str(score[1]))
        self.logger.log('\nprecision: ')
        self.logger.log(str(score[2]))
        self.logger.log('\nrecall: ')
        self.logger.log(str(score[3]))
        self.logger.log('\nf1_score: ')
        self.logger.log(str(score[4]))


    def get_model(self):
        return self.model

    # @staticmethod
    # def get_random_parameters():
    #     params = {}
    #     params['units'] = random.randrange(100, 1000, 100)
    #     return params

    @staticmethod
    def get_exhaustive_parameters():
        params_list = []

        unit_counts = []
        # for i in range(0, 12):
        for i in range(7, 8):
            unit_counts.append(int(pow(2, i)))

        for unit_count in unit_counts:
            # for window_size in range(1, 6):
            for window_size in range(1, 2):
                params = {}
                params['units'] = unit_count
                params['window_size'] = window_size
                params_list.append(params)
        
        return params_list

    @staticmethod
    def log_parameters(l, params):
        l.log('units: ' + str(params['units']) + '\n')
        l.log('window_size: ' + str(params['window_size']) + '\n')

def precision(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(y_true * y_pred))
    c2 = K.sum(K.round(y_pred))
    c3 = K.sum(K.round(y_true))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    return precision

def recall(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(y_true * y_pred))
    c2 = K.sum(K.round(y_pred))
    c3 = K.sum(K.round(y_true))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many relevant items are selected?
    recall = c1 / c3

    return recall

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(y_true * y_pred))
    c2 = K.sum(K.round(y_pred))
    c3 = K.sum(K.round(y_true))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3

    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score