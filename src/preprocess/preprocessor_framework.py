from src.preprocess.corpus_reader import CorpusReader 

from src.preprocess.lstm_baseline_preprocessor import LstmBaselinePreprocessor
from src.preprocess.feedforward_preprocessor import FeedforwardPreprocessor

import numpy as np
import os.path

# import argparse

VOWEL_TABLE = {'a': ['a', 'á'], 'e': ['e', 'é'], 'i': ['i', 'í'], 'o': ['o', 'ó', 'ö', 'ő'], 'u': ['u', 'ú', 'ü', 'ű']}
RESOURCE_DIRECTORY = 'res'

class PreprocessorFramework:

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    # def read_corpus(self, vowel, count):
    #     corpus = open(os.path.join(RESOURCE_DIRECTORY, "corpus"), encoding="utf8")
    #     words = []
    #     accent_counter = {}

    #     accents = VOWEL_TABLE[vowel]
    #     for accent in accents:
    #         accent_counter[accent] = 0

    #     for i in range(4):
    #         next(corpus)

    #     line_counter = 0
    #     for line in corpus:
    #         line_counter += 1
    #         enough = True

    #         splits = line.split()
    #         if splits != []:
    #             word = splits[0]
    #             words.append(word)

    #             for accent in accents:
    #                 accent_counter[accent] += word.count(accent)
    #                 if accent_counter[accent] < count:
    #                     enough = False

    #             if enough:
    #                 return words

    #         if line_counter % 10000 == 0:
    #             print('line: ' + str(line_counter))
    #             for accent in accents:
    #                 print("\t" + accent + ": " + str(accent_counter[accent]))

    #     print("ERROR: Corpus has run out of words!")
    #     return words

    def process(self, count, window_size, vowel=None):

        if vowel:
            words = CorpusReader.read_words(vowel, count)
            px, py = self.create_preprocessor(count, window_size, vowel).make_windows(words)
            np.savez(self.create_path(window_size, vowel), x=px, y=py)
        else:
            for vowel in VOWEL_TABLE.keys():
                words = CorpusReader.read_words(vowel, count)
                px, py = self.create_preprocessor(count, window_size, vowel).make_windows(words)
                np.savez(self.create_path(window_size, vowel), x=px, y=py)
                
            file = open(os.path.join(RESOURCE_DIRECTORY, 'prepared', self.preprocessor, str(window_size), '_count.txt'), 'w')
            file.write(str(count))

    def create_preprocessor(self, count, window_size, vowel):
        if self.preprocessor == 'lstm_baseline':
            return LstmBaselinePreprocessor(count, window_size, vowel)
        elif self.preprocessor == 'feedforward':
            return FeedforwardPreprocessor(count, window_size, vowel)

    def create_path(self, window_size, vowel):
        parent_path = os.path.join(RESOURCE_DIRECTORY, 'prepared', self.preprocessor, str(window_size))
        os.makedirs(parent_path, exist_ok=True)
        return os.path.join(parent_path, vowel)

# parser = argparse.ArgumentParser()
# parser.add_argument("preprocessor")
# parser.add_argument("count")
# parser.add_argument("window_size")
# parser.add_argument("--vowel")

# args = parser.parse_args()
# preprocessor = args.preprocessor
# count = int(args.count)
# window_size = int(args.window_size)
# vowel = args.vowel

# framework = PreprocessorFramework(preprocessor)
# framework.process(count, window_size, vowel)