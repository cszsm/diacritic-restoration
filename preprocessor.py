import lstm_baseline_preprocessor as processor

import numpy as np

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
    # print(words[1800:])
    px, py = preprocessor(words, window_size, vowel)
    np.savez("prepared_" + vowel, x=px, y=py)
    # return preprocessor(words, window_size, vowel)

process(processor.make_windows, 10000, 1, 'a')
