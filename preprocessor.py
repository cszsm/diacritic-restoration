import lstm_baseline_preprocessor

import numpy as np

import argparse

def read_corpus(count):
    corpus = open("corpus")
    words = []

    for i in range(4):
        next(corpus)

    for line in corpus:
        splits = line.split()
        if splits != []:
            words.append(splits[0])
            if count < 0:
                break
            count -= 1

    return words

def process(preprocessor, count, window_size, vowel):
    words = read_corpus(count)
    px, py = preprocessor(words, window_size, vowel)
    np.savez("prepared_" + vowel, x=px, y=py)

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
    preprocess = lstm_baseline_preprocessor.make_windows

process(preprocess, count, window_size, vowel)
