import lstm_baseline_preprocessor

import numpy as np
import os.path

import argparse

VOWEL_TABLE = {'a': ['a', 'á'], 'e': ['e', 'é'], 'i': ['i', 'í'], 'o': ['o', 'ó', 'ö', 'ő'], 'u': ['u', 'ú', 'ü', 'ű']}
RESOURCE_DIRECTORY = '../../res/'

def read_corpus(vowel, count):
    corpus = open(os.path.join(RESOURCE_DIRECTORY, "corpus"))
    words = []
    accent_counter = {}

    accents = VOWEL_TABLE[vowel]
    for accent in accents:
        accent_counter[accent] = 0

    for i in range(4):
        next(corpus)

    line_counter = 0
    for line in corpus:
        line_counter += 1
        enough = True

        splits = line.split()
        if splits != []:
            word = splits[0]
            words.append(word)

            for accent in accents:
                accent_counter[accent] += word.count(accent)
                if accent_counter[accent] < count:
                    enough = False

            if enough:
                return words

        if line_counter % 10000 == 0:
            print('line: ' + str(line_counter))
            for accent in accents:
                print("\t" + accent + ": " + str(accent_counter[accent]))

    print("ERROR: Corpus has run out of words!")
    return words

def process(preprocess):
    words = read_corpus(vowel, count)
    px, py = preprocess(words)
    np.savez(os.path.join(RESOURCE_DIRECTORY, "prepared_" + vowel), x=px, y=py)

parser = argparse.ArgumentParser()
parser.add_argument("preprocessor")
parser.add_argument("count")
parser.add_argument("window_size")
parser.add_argument("vowel")

args = parser.parse_args()
preprocessor = args.preprocessor
count = int(args.count)
window_size = int(args.window_size)
vowel = args.vowel

preprocess = {}
if preprocessor == 'lstm_baseline':
    preprocess = lstm_baseline_preprocessor.LstmBaselinePreprocessor(count, window_size, vowel)

process(preprocess.make_windows)
