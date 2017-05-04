"""Preprocess data for the baseline lstm network"""
from collections import deque
from sklearn.feature_extraction import DictVectorizer
import numpy as np

import preprocessor_common as common

VOWELS = "aáeéiíoóöőuúüű"
VOWEL_TABLE = {'a': ['a', 'á'], 'e': ['e', 'é'], 'i': ['i', 'í'], 'o': ['o', 'ó', 'ö', 'ő'], 'u': ['u', 'ú', 'ü', 'ű']}
VECTORIZER = DictVectorizer()

class LstmBaselinePreprocessor:

    accent_counter = {}

    def __init__(self, count, window_size, vowel):
        self.count = count
        self.window_size = window_size
        self.vowel = vowel

    def make_windows_from_word(self, word):
        windows = []
        accents = []

        sliding_window = deque((), self.window_size * 2 + 1)

        for character in word[:sliding_window.maxlen - 1]:
            sliding_window.append(character)

        for character in word[sliding_window.maxlen - 1:]:
            sliding_window.append(character)

            if (sliding_window[self.window_size] in VOWEL_TABLE[self.vowel]) and (self.accent_counter[sliding_window[self.window_size]] < self.count):
                normalized_list = list(common.deaccentize_list(list(sliding_window)))
                transformed_list = common.transform_list(normalized_list)
                transformed_accents = sliding_window[self.window_size]

                self.accent_counter[sliding_window[self.window_size]] += 1

                windows.append(transformed_list)
                accents.append(common.transform_accent(transformed_accents))

        return windows, accents


    def make_windows_from_text(self, text):
        windows = []
        accents = []

        for accent in VOWEL_TABLE[self.vowel]:
            self.accent_counter[accent] = 0

        word_counter = 0
        for word in text:
            word_counter += 1

            normalized_word = common.normalize_text(word.lower())
            padded_word = common.pad_word(normalized_word, self.window_size)

            new_windows, new_accents = self.make_windows_from_word(padded_word)

            windows += new_windows
            accents += new_accents

            if word_counter % 100 == 0:
                print('word: ' + str(word_counter))
                for accent in VOWEL_TABLE[self.vowel]:
                    print("\t" + accent + ": " + str(self.accent_counter[accent]))

        return windows, accents

    def make_windows(self, text):
        common.fit_encoders()
        return self.make_windows_from_text(text)


    @staticmethod
    def preprocess(text, window_size):
        common.fit_encoders()
        windows = {}

        for vowel in VOWEL_TABLE.keys():
            windows[vowel] = []

            for word in text:
                normalized_word = common.normalize_text(word.lower())
                padded_word = common.pad_word(normalized_word, window_size)

                new_windows = LstmBaselinePreprocessor.helper(padded_word, window_size, vowel)

                windows[vowel] += new_windows

        return windows

    @staticmethod
    def helper(word, window_size, vowel):
        windows = []

        sliding_window = deque((), window_size * 2 + 1)

        for character in word[:sliding_window.maxlen - 1]:
            sliding_window.append(character)

        for character in word[sliding_window.maxlen - 1:]:
            sliding_window.append(character)

            if (sliding_window[window_size] in VOWEL_TABLE[vowel]):
                normalized_list = list(common.deaccentize_list(list(sliding_window)))
                transformed_list = common.transform_list(normalized_list)

                windows.append(transformed_list)

        return windows