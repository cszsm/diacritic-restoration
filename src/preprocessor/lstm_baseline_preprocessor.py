"""Preprocess data for the baseline lstm network"""
from collections import deque
from sklearn.feature_extraction import DictVectorizer
import numpy as np

import preprocessor_common as common

VOWELS = "aáeéiíoóöőuúüű"
# VOWEL_TABLE = {"a": ["á"], "e": ["é"], "i": ["í"],
#                "o": ["ó", "ö", "ő"], "u": ["ú", "ü", "ű"]}
VOWEL_TABLE = {'a': ['a', 'á'], 'e': ['e', 'é'], 'i': ['i', 'í'], 'o': ['o', 'ó', 'ö', 'ő'], 'u': ['u', 'ú', 'ü', 'ű']}
VECTORIZER = DictVectorizer()

class LstmBaselinePreprocessor:

    accent_counter = {}

    def make_windows_from_word(self, word, count, window_size, vowel):
        windows = []
        accents = []

        sliding_window = deque((), window_size * 2 + 1)

        for accent in VOWEL_TABLE[vowel]:
            self.accent_counter[accent] = 0

        for character in word[:sliding_window.maxlen - 1]:
            sliding_window.append(character)

        for character in word[sliding_window.maxlen - 1:]:
            sliding_window.append(character)

            if (sliding_window[window_size] in VOWEL_TABLE[vowel]) and (self.accent_counter[vowel] < count):
                normalized_list = list(common.deaccentize_list(list(sliding_window)))
                transformed_list = common.transform_list(normalized_list)
                transformed_accents = sliding_window[window_size]

                self.accent_counter[vowel] += 1

                windows.append(transformed_list)
                accents.append(common.transform_accent(transformed_accents))

        return windows, accents


    def make_windows_from_text(self, text, count, window_size, vowel):
        windows = []
        accents = []

        for word in text:
            normalized_word = common.normalize_text(word)
            padded_word = common.pad_word(normalized_word.lower(), window_size)

            new_windows, new_accents = self.make_windows_from_word(
                padded_word, count, window_size, vowel)

            windows += new_windows
            accents += new_accents

        return windows, accents

    def make_windows(self, text, count, window_size, vowel):
        common.fit_encoders()
        return self.make_windows_from_text(text, count, window_size, vowel)
