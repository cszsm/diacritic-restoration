import random
import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

import keras.backend as K
import preprocessor.lstm_baseline_preprocessor as preprocessor

from math import pow

VOWEL_TABLE = {'a': ['a', 'á'], 'e': ['e', 'é'], 'i': ['i', 'í'], 'o': ['o', 'ó', 'ö', 'ő'], 'u': ['u', 'ú', 'ü', 'ű']}

class Network:

    vowel = ''

    def __init__(self, params, vowel, logger):

        self.vowel = vowel
        self.units = params['units']
        self.window_size = params['window_size']

        self.logger = logger

        self.model = Sequential()
        self.model.add(LSTM(self.units, return_sequences=False, input_shape=(self.window_size * 2, 30)))

        self.output_length = 2
        if self.vowel in ['o', 'u']:
            self.output_length = 4

        self.model.add(Dense(self.output_length))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    def run(self, train_x, test_x, train_y, test_y):

        self.logger.log('\nvowel: ' + self.vowel)

        early_stopping = EarlyStopping(monitor='loss', patience=0)
        self.model.fit(train_x, train_y, batch_size=32, epochs=100, callbacks=[early_stopping], verbose=3)
        score = self.model.evaluate(test_x, test_y, batch_size=32)

        self.logger.log('loss: ' + str(score[0]))
        self.logger.log('accuracy: ' + str(score[1]))

        predictions = self.model.predict_classes(test_x)
        actuals = test_y

        for i in range(self.output_length):
            self.logger.log('\n' + VOWEL_TABLE[self.vowel][i])
            tp = 0
            fp = 0
            fn = 0


            for j in range(len(predictions)):
                actual = np.array(actuals[j]).argmax(axis=0)
                prediction = predictions[j]
                if actual == i:
                    if prediction == actual:
                        tp += 1
                if prediction == i:
                    if prediction != actual:
                        fp += 1
                if actual == i:
                    if prediction != actual:
                        fn += 1

            print('tp: ' + str(tp))
            print('fp: ' + str(fp))
            print('fn: ' + str(fn))

            if tp + fp == 0:
                precision = -1
            else:
                precision = tp / (tp + fp)
            print('precision: ' + str(precision))

            if tp + fn == 0:
                recall = -1
            else:
                recall = tp / (tp + fn)
            print('recall: ' + str(recall))

            if precision == -1 or recall == -1 or precision + recall == 0:
                fscore = -1
            else:
                fscore = 2 * ((precision * recall) / (precision + recall))
            self.logger.log('fscore: ' + str(fscore))


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
        for i in range(10, 11):
        # for i in range(0, 1):
            unit_counts.append(int(pow(2, i)))

        for unit_count in unit_counts:
            # for window_size in range(1, 6):
            for window_size in range(3, 4):
                params = {}
                params['units'] = unit_count
                params['window_size'] = window_size
                params_list.append(params)

        return params_list

    @staticmethod
    def log_parameters(l, params):
        l.log('units: ' + str(params['units']))
        l.log('window_size: ' + str(params['window_size']))
