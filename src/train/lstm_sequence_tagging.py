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

from src.accentizer import _decode_predictions_to_string

import os.path
from src.f_score import FScore

INPUT_DIM = 30
OUTPUT_DIM = 39
SEQUENCE_LEN = 600
EPOCHS = 3


class Network:
    def __init__(self, logger, bidirectional):

        self.logger = logger

        self.model = Sequential()
        self.model.add(
            Masking(mask_value=0., input_shape=(SEQUENCE_LEN, INPUT_DIM)))
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
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])

        print(self.model.summary())

    def train(self):

        # train_x, valid_test_x, train_y, valid_test_y = train_test_split(
        #     data_x, data_y, test_size=0.4)
        # valid_x, test_x, valid_y, test_y = train_test_split(
        #     valid_test_x, valid_test_y, test_size=0.5)

        with open(os.path.join('res', 'sentences'), encoding='utf8') as f:
            early_stopping = EarlyStopping(monitor='val_loss', patience=0)

            batch_size = 128
            # steps_per_epoch = int(len(train_x) / batch_size)
            steps_per_epoch = 75

            self.model.fit_generator(
                next_batch(f, steps_per_epoch, 'train'),
                steps_per_epoch,
                epochs=EPOCHS,
                callbacks=[early_stopping],
                verbose=1,
                validation_data=next_batch(f, steps_per_epoch, 'valid'),
                validation_steps=steps_per_epoch)

            score = self.model.evaluate_generator(
                next_batch(f, batch_size, 'test'), steps_per_epoch)
            print(score)

            # TODO: remove
            texts = ['arvizturo', 'tukorfurogep']
            characters, _ = process(texts)
            padded = pad_sequences(
                characters, maxlen=SEQUENCE_LEN, padding='post', value=0.)
            predictions = self.model.predict(np.array(padded))
            example = _decode_predictions_to_string(predictions[0])
            print(example)
            print(len(example))

            test_sentences = []
            for i in range(200):
                test_sentences.append(f.readline().rstrip('\n'))

            test_x, test_y = process(test_sentences)

            _calculate_fscores(self.model, test_x, test_y)

    def get_model(self):
        return self.model


def _calculate_fscores(model, test_x, test_y):

    i = 0
    f_scores = {}
    vowels = 'aáeéiíoóöőuúüű'
    for vowel in vowels:
        f_scores[vowel] = FScore()

    for batch_x, batch_y in next_batch_for_fscore(test_x, test_y, 100):
        i += 100

        if i > len(test_x):
            break

        predictions = model.predict_classes(batch_x)

        # print('predictions')
        # print(predictions)
        # print('sample_y')
        # print(batch_y)
        # for x, y in zip(predictions, batch_y):
        #     print(_decode_predictions_to_string(x))
        #     print(_decode_predictions_to_string(y))

        for x, y in zip(predictions, batch_y):
            for vowel in vowels:
                current_score = _calculate_fscore(
                    HungarianEncoder().transform_label(vowel), x, y)
                f_scores[vowel] += current_score

    # for sample_x, sample_y in zip(test_x, test_y):
    #     i += 1

    #     padded = pad_sequences(
    #         [sample_x], maxlen=SEQUENCE_LEN, padding='post', value=0.)
    #     # print('padded')
    #     # print(padded)
    #     np_padded = np.array(padded)
    #     # print('np_padded')
    #     # print(np_padded)
    #     predictions = model.predict_classes(np_padded)
    #     # predictions = model.predict(np_padded)
    #     # print('predictions')
    #     # print(predictions)
    #     # print('sample_y')
    #     # print(sample_y)
    #     # print(_decode_predictions_to_string([sample_y]))
    #     for vowel in vowels:
    #         current_score = _calculate_fscore(
    #             HungarianEncoder().transform_label(vowel), predictions[0],
    #             sample_y)
    #         f_scores[vowel] += current_score

        print(i)

    for vowel in vowels:
        print(vowel)
        print(f_scores[vowel])

    with open('log.txt', 'w') as f:
        for vowel in vowels:
            f.write(vowel)
            f.write('\n')
            f.write(str(f_scores[vowel]))
            f.write('\n\n')


# TODO: rename
def _calculate_fscore(character, predictions, actuals):
    '''Calculates the number of true positives, false positives and false negatives for the given character'''
    # self.logger.log('\n' + VOWEL_TABLE[self.vowel][i])
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # print(character)

    # TODO zip
    for j in range(len(actuals)):
        # print('predictions[' + str(j) + ']')
        # print(predictions[j])

        actual = np.array(actuals[j]).argmax(axis=0)
        # print('actuals[' + str(j) + ']')
        # print(actual)
        prediction = predictions[j]
        if actual == character:
            if prediction == actual:
                true_positives += 1
            else:
                false_negatives += 1
        elif prediction == character and prediction != actual:
            false_positives += 1

    return FScore(true_positives, false_positives, false_negatives)
    # self.logger.log('fscore: ' + str(fscore))


def next_batch_for_fscore(data_x, data_y, batch_size):

    batch_x = np.zeros((batch_size, SEQUENCE_LEN, INPUT_DIM))
    batch_y = np.zeros((batch_size, SEQUENCE_LEN, OUTPUT_DIM))

    counter = 0
    while True:
        bx = data_x[counter:counter + batch_size]
        by = data_y[counter:counter + batch_size]

        padded_x = pad_sequences(
            bx, maxlen=SEQUENCE_LEN, padding='post', value=0.)
        padded_y = pad_sequences(
            by, maxlen=SEQUENCE_LEN, padding='post', value=0.)

        counter += batch_size
        yield padded_x, padded_y


def next_batch(f, batch_size, dataset):

    batch_x = np.zeros((batch_size, SEQUENCE_LEN, INPUT_DIM))
    batch_y = np.zeros((batch_size, SEQUENCE_LEN, OUTPUT_DIM))

    count = {}
    count['train'] = 0
    count['valid'] = 0
    count['test'] = 0

    while True:
        for i in range(batch_size):
            count[dataset] += 1

            # index = random.randrange(len(data_x))

            sentence = f.readline().rstrip('\n')
            # print(str(j) + ' ' + dataset)
            # print(sentence)
            x, y = process([sentence])

            # TODO: hack
            if len(x) == 0:
                i -= 1
                continue

            bx = pad_sequences(
                x, maxlen=SEQUENCE_LEN, padding='post', value=0.)
            by = pad_sequences(
                y, maxlen=SEQUENCE_LEN, padding='post', value=0.)

            # print('')
            # print('________')
            # print(_decode_predictions_to_string(bx[0]))
            # print(_decode_predictions_to_string(by[0]))
            # print('________')

            batch_x[i] = bx[0]
            batch_y[i] = by[0]

        # print('')

        # print('____x____')
        # print(x)
        # print(bx[0][0])

        # print('train\t' + str(count['train']))
        # print('valid\t' + str(count['valid']))
        # print('test\t' + str(count['test']))

        # print('batch')
        # print(batch_x)
        # print(batch_x[0])
        # print(_decode_predictions_to_string(batch_x[0]))
        # print(batch_x[0])
        # print(_decode_predictions_to_string(batch_y[0]))
        yield batch_x, batch_y