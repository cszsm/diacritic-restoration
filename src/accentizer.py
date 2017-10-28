# import os
import os.path

from keras.models import load_model
# import tensorflow as tf
import numpy as np

# import src.preprocess.feedforward_preprocessor as feedforward_preprocessor
import src.preprocess.lstm_baseline_preprocessor as lstm_baseline_preprocessor
from src.preprocess.lstm_sequence_tagging import process

from src.preprocess import common

from src.preprocess.character_encoder import HungarianEncoder

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


# def accentize_with_feedforward(text, model_id):
#     windows = feedforward_preprocessor.FeedforwardPreprocessor.preprocess([text], 4)
#     accents = {}

#     for vowel in VOWEL_TABLE.keys():
#         path = os.path.join(MODEL_PATH, 'feedforward', model_id, vowel + '.model')

#         tf.reset_default_graph()
#         with tf.Session() as sess:
#             saver = tf.train.import_meta_graph(path + '.meta')
#             saver.restore(sess, path)

#             accents[vowel] = sess.run('output:0', feed_dict={'n_input:0': windows[vowel]})

#     accentized_text = accentize_with_accents(text, accents)

#     print('feedforward: ' + accentized_text)


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

    sentence_length = len(text)
    if sentence_length <= 600:

        d = 600 - sentence_length
        for i in range(d):
            text += '_'

    characters, _ = process([text])

    path = os.path.join(MODEL_PATH, 'lstm_sequence_tagging', 'trained.model')
    model = load_model(path)
    predictions = model.predict(np.array(characters))

    # predictions to print
    preds_to_print = predictions[0][0]
    rounded_preds = np.around(preds_to_print, 2)
    print(rounded_preds)

    encoder = HungarianEncoder()
    accentized = ''

    for prediction in predictions[0]:
        character = encoder.inverse_transform(prediction)
        accentized += character

    return accentized
