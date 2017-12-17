# import os
import os.path

from keras.models import load_model
import tensorflow as tf
import numpy as np

# import src.preprocess.feedforward_preprocessor as feedforward_preprocessor
import src.preprocess.lstm_baseline_preprocessor as lstm_baseline_preprocessor
from src.preprocess.lstm_sequence_tagging import process

from keras.preprocessing.sequence import pad_sequences

from src.preprocess import common

from src.preprocess.character_encoder import HungarianEncoder

from src.preprocess.feedforward import process_for_accentize

# import argparse

VOWEL_TABLE = {
    'a': ['a', 'á'],
    'e': ['e', 'é'],
    'i': ['i', 'í'],
    'o': ['o', 'ó', 'ö', 'ő'],
    'u': ['u', 'ú', 'ü', 'ű']
}
MODEL_PATH = os.path.join('models')


def accentize(text, network_type, units, window_size):
    if network_type == 'feedforward':
        pass
        # accentize_with_feedforward(text, model_id)
    if network_type == 'lstm_baseline':
        return accentize_with_lstm_baseline(text, units, window_size)


def accentize_with_feedforward(text, args):
    windows = process_for_accentize(4, [text])
    accents = {}

    for vowel in VOWEL_TABLE.keys():

        if len(windows[vowel]) == 0:
            continue

        path = os.path.join(MODEL_PATH, 'feedforward', args, vowel + '.model')

        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(path + '.meta')
            saver.restore(sess, path)

            accents[vowel] = sess.run(
                'output:0', feed_dict={
                    'n_input:0': windows[vowel]
                })

    accentized_text = accentize_with_accents(text, accents)

    return accentized_text


def accentize_with_lstm_baseline(text, units, window_size):
    windows = lstm_baseline_preprocessor.LstmBaselinePreprocessor.preprocess(
        [text], window_size)
    accents = {}

    for vowel in VOWEL_TABLE.keys():
        if len(windows[vowel]) == 0:
            continue

        model = load_model(
            os.path.join(MODEL_PATH, 'lstm_baseline',
                         str(units), str(window_size), vowel + '.model'))
        accents[vowel] = model.predict(np.array(windows[vowel]))

    accentized_text = accentize_with_accents(text, accents)

    print('lstm_baseline: ' + accentized_text)


def decode_accent(vowel, coded_accent):
    '''Returns the vowel corresponding to the one-hot encoded vowel'''
    i = np.array(coded_accent).argmax(axis=0)
    return VOWEL_TABLE[vowel][i]


def accentize_with_accents(text, accents):
    '''Puts the accents on a deaccentized text'''
    vowel_counter = {}
    for vowel in VOWEL_TABLE.keys():
        vowel_counter[vowel] = 0

    accentized_text = ''
    for c in text.lower():
        if c in VOWEL_TABLE.keys():
            accentized_text += decode_accent(c, accents[c][vowel_counter[c]])
            vowel_counter[c] += 1
        else:
            accentized_text += c

    return accentized_text


def accentize_with_lstm_sequence_tagging(text):
    '''Accentizes text with the sequence tagging LSTM network'''

    sequence_len = 300

    path = os.path.join(MODEL_PATH, 'bidirectional_lstm_sequence_tagging',
                        'trained.model')
    model = load_model(path)
    encoder = HungarianEncoder()

    extracted_text, multiple_spaces = _extract_spaces(text)
    characters, _ = process([extracted_text])
    characters = pad_sequences(
        characters, maxlen=sequence_len, padding='post', value=0.)
    predictions = model.predict(np.array(characters))

    accentized = ''
    for prediction in predictions[0]:
        character = encoder.inverse_transform(prediction)
        accentized += character

    restored_text = _restore_spaces(accentized, multiple_spaces)
    denormalized = _denormalize(text, restored_text)

    return denormalized


def _decode_predictions_to_string(predictions):
    '''Create readable text from predictions'''

    encoder = HungarianEncoder()
    accentized = ''

    for prediction in predictions:
        character = encoder.inverse_transform(prediction)
        accentized += character

    return accentized


def _extract_spaces(text):
    '''Removes unnecessary spaces where more than one occur'''

    multiple_spaces = []
    extracted_text = ''

    count = 0

    for character in text:
        if character == ' ':
            count += 1
        else:
            if count > 0:
                if count > 1:
                    multiple_spaces.append((len(extracted_text), count))

                extracted_text += ' '
                extracted_text += character
                count = 0
            else:
                extracted_text += character

    return extracted_text, multiple_spaces


def _restore_spaces(text, multiple_spaces):
    '''Restores spaces in the text'''

    i = 0
    restored_text = ''

    for index, count in multiple_spaces:
        restored_text += text[i:index]
        restored_text += ' ' * count
        i = index + 1

    restored_text += text[i:]

    return restored_text


def _denormalize(original_text, accentized_text):
    '''Restore capitalization and punctuations in accentized text based on the original text'''

    denormalized = ''

    for original, accentized in zip(original_text, accentized_text):
        if original.lower() in VOWEL_TABLE.keys():
            if accentized in VOWEL_TABLE[original.lower()]:
                if original.isupper():
                    denormalized += accentized.upper()
                else:
                    denormalized += accentized
            else:
                denormalized += original
        else:
            denormalized += original

    return denormalized
