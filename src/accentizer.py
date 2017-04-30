from keras.models import load_model
import os.path
import numpy as np

import preprocessor.lstm_baseline_preprocessor as lstm_baseline_preprocessor

# VOWELS = ['a', 'e', 'i', 'o', 'u']
VOWEL_TABLE = {'a': ['a', 'á'], 'e': ['e', 'é'], 'i': ['i', 'í'], 'o': ['o', 'ó', 'ö', 'ő'], 'u': ['u', 'ú', 'ü', 'ű']}
MODEL_PATH = '../models'

def accentize(text, model_id):
    windows = lstm_baseline_preprocessor.LstmBaselinePreprocessor.preprocess([text], 1)
    accents = {}

    # for vowel in VOWEL_TABLE.keys():
    #     print(vowel)
    #     print(windows[vowel])

    for vowel in VOWEL_TABLE.keys():
        model = load_model(os.path.join(MODEL_PATH, 'lstm_baseline', model_id, vowel + '.model'))
        accents[vowel] = model.predict(np.array(windows[vowel]))

    # for vowel in VOWEL_TABLE.keys():
    #     print(vowel)
    #     print(accents[vowel])

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

    print(accentized_text)

        
def decode_accent(vowel, coded_accent):
    i = np.array(coded_accent).argmax(axis=0)
    return VOWEL_TABLE[vowel][i]

accentize('Akar eleg ozek izu', '2017-04-30_0')