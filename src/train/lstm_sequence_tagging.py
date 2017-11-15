from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Masking, TimeDistributed, Bidirectional
from keras.callbacks import EarlyStopping

from keras.preprocessing.sequence import pad_sequences

import random

from sklearn.model_selection import train_test_split
import numpy as np

from src.accentizer import _decode_predictions_to_string

from src.preprocess.character_encoder import EnglishEncoder, HungarianEncoder

from src.preprocess.lstm_sequence_tagging import process

INPUT_DIM = 30
OUTPUT_DIM = 39
SEQUENCE_LEN = 100


class Network:
    def __init__(self, logger, bidirectional):

        self.logger = logger

        self.model = Sequential()
        self.model.add(
            Masking(mask_value=0., input_shape=(SEQUENCE_LEN, INPUT_DIM)))
        # self.model.add(LSTM(100, input_shape=(600, 31), return_sequences=True))
        if bidirectional:
            self.model.add(Bidirectional(LSTM(100, return_sequences=True)))
            self.model.add(Bidirectional(LSTM(100, return_sequences=True)))
            self.model.add(Bidirectional(LSTM(100, return_sequences=True)))
        else:
            self.model.add(LSTM(100, return_sequences=True))
            self.model.add(LSTM(100, return_sequences=True))
            self.model.add(LSTM(100, return_sequences=True))
        self.model.add(TimeDistributed(Dense(OUTPUT_DIM)))
        self.model.add(Activation('softmax'))

        self.model.compile(
            # loss='binary_crossentropy',
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

        print(self.model.summary())

    def train(self, train_x, test_x, train_y, test_y):

        early_stopping = EarlyStopping(monitor='val_loss', patience=0)
        # self.model.fit(
        #     train_x,
        #     train_y,
        #     batch_size=32,
        #     epochs=100,
        #     callbacks=[early_stopping],
        #     verbose=1,
        #     validation_split=0.4)

        t_x, v_x, t_y, v_y = train_test_split(train_x, train_y, test_size=0.4)

        batch_size = 128
        steps_per_epoch = len(t_x) / batch_size
        print(steps_per_epoch)
        steps_per_epoch = int(steps_per_epoch)
        print(steps_per_epoch)
        # steps_per_epoch = 20
        self.model.fit_generator(
            next_batch(t_x, t_y, steps_per_epoch),
            steps_per_epoch,
            epochs=10,
            callbacks=[early_stopping],
            verbose=1,
            validation_data=next_batch(v_x, v_y, steps_per_epoch),
            validation_steps=steps_per_epoch)
        score = self.model.evaluate_generator(
            next_batch(test_x, test_y, batch_size), steps_per_epoch)
        print(score)

        texts = ['arvizturo', 'tukorfurogep']

        characters, _ = process(texts)
        padded = pad_sequences(
            characters, maxlen=SEQUENCE_LEN, padding='post', value=0.)

        predictions = self.model.predict(np.array(padded))

        # print(predictions)
        example = _decode_predictions_to_string(predictions)
        print(example)
        print(len(example))

    def get_model(self):
        return self.model


# TODO move it to e.g. common
# or not, this is a generator
def next_batch(data_x, data_y, batch_size):

    batch_x = np.zeros((batch_size, SEQUENCE_LEN, INPUT_DIM))
    batch_y = np.zeros((batch_size, SEQUENCE_LEN, OUTPUT_DIM))

    while True:
        for i in range(batch_size):

            index = random.randrange(len(data_x))

            if len(data_x[index]) == 0:
                i -= 1
                continue

            # print(data_x[index])
            # print(len(data_x[index]))
            # print(_decode_predictions_to_string([data_x[index]]))
            # print(_decode_predictions_to_string([data_y[index]]))

            # print('data: ', np.shape(data_x[index]))
            bx = pad_sequences(
                [data_x[index]], maxlen=SEQUENCE_LEN, padding='post', value=0.)
            # print('padded: ', np.shape(bx))
            by = pad_sequences(
                [data_y[index]], maxlen=SEQUENCE_LEN, padding='post', value=0.)

            batch_x[i] = bx[0]
            batch_y[i] = by[0]

        yield batch_x, batch_y

        # print('most ' + str(i))
        # # data = [data_x[index]], [data_y[index]]
        # array_x = np.ndarray(shape=(1, 600, 30), buffer=data_x[index])
        # array_y = np.ndarray(shape=(1, 600, 39), buffer=data_y[index])
        # yield (array_x, array_y)

    # batch_x, batch_y = [], []

    # indices = random.sample(range(len(data_x)), 32)

    # for i in indices:
    #     batch_x.append(data_x[i])
    #     batch_y.append(data_y[i])

    # yield ([batch_x], [batch_y])
