import os
import os.path

from keras.models import load_model
import tensorflow as tf
import numpy as np

import preprocessor.feedforward_preprocessor as feedforward_preprocessor
import preprocessor.lstm_baseline_preprocessor as lstm_baseline_preprocessor

import argparse

VOWEL_TABLE = {'a': ['a', 'á'], 'e': ['e', 'é'], 'i': ['i', 'í'], 'o': ['o', 'ó', 'ö', 'ő'], 'u': ['u', 'ú', 'ü', 'ű']}
MODEL_PATH = '../models'

def accentize(text, network_type, model_id):
    if network_type == 'feedforward':
        accentize_with_feedforward(text, model_id)
    if network_type == 'lstm_baseline':
        return accentize_with_lstm_baseline(text, model_id)

def accentize_with_feedforward(text, model_id):
    windows = feedforward_preprocessor.FeedforwardPreprocessor.preprocess([text], 4)
    accents = {}

    for vowel in VOWEL_TABLE.keys():
        path = os.path.join(MODEL_PATH, 'feedforward', model_id, vowel + '.model')
 
        tf.reset_default_graph()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(path + '.meta')
            saver.restore(sess, path)

            accents[vowel] = sess.run('output:0', feed_dict={'n_input:0': windows[vowel]})

    accentized_text = accentize_with_accents(text, accents)

    print('feedforward: ' + accentized_text)
    

def accentize_with_lstm_baseline(text, model_id):
    windows = lstm_baseline_preprocessor.LstmBaselinePreprocessor.preprocess([text], 1)
    accents = {}

    for vowel in VOWEL_TABLE.keys():
        model = load_model(os.path.join(MODEL_PATH, 'lstm_baseline', model_id, vowel + '.model'))
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


parser = argparse.ArgumentParser()
parser.add_argument('--network_type', help='can be \'feedforward\' or \'lstm_baseline\'')
parser.add_argument('--model_id')
parser.add_argument('--text')

args = parser.parse_args()

if args.network_type is None or args.model_id is None or args.text is None:
    print('Default accentizing')

    files = os.listdir(os.path.join(MODEL_PATH, 'feedforward'))
    for f in files:
        print(f)
        accentize('arvizturo tukorfurogep', 'feedforward', f)
    
    files = os.listdir(os.path.join(MODEL_PATH, 'lstm_baseline'))
    for f in files:
        print(f)
        accentize('arvizturo tukorfurogep', 'lstm_baseline', f)
else:
    accentize(args.text, args.network_type, args.model_id)
