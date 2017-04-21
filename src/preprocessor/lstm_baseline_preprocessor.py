"""Preprocess data for the baseline lstm network"""
from collections import deque
from sklearn.feature_extraction import DictVectorizer
import numpy as np

import preprocessor_common as common

VOWELS = "aáeéiíoóöőuúüű"
VOWEL_TABLE = {"a": ["á"], "e": ["é"], "i": ["í"],
               "o": ["ó", "ö", "ő"], "u": ["ú", "ü", "ű"]}
VECTORIZER = DictVectorizer()


def make_windows_from_word(word, window_size, vowel):
    windows = []
    accents = []

    sliding_window = deque((), window_size * 2 + 1)

    for character in word[:sliding_window.maxlen - 1]:
        sliding_window.append(character)

    for character in word[sliding_window.maxlen - 1:]:
        sliding_window.append(character)

        if (sliding_window[window_size] == vowel) \
                or (sliding_window[window_size] in VOWEL_TABLE[vowel]):
            normalized_list = list(common.deaccentize_list(list(sliding_window)))
            transformed_list = common.transform_list(normalized_list)
            transformed_accents = sliding_window[window_size]

            windows.append(transformed_list)
            accents.append(common.transform_accent(transformed_accents))

    return windows, accents


def make_windows_from_text(text, window_size, vowel):
    windows = []
    accents = []

    for word in text:
        normalized_word = common.normalize_text(word)
        padded_word = common.pad_word(normalized_word.lower(), window_size)

        new_windows, new_accents = make_windows_from_word(
            padded_word, window_size, vowel)

        windows += new_windows
        accents += new_accents

    return windows, accents

def make_windows(text, window_size, vowel):
    common.fit_encoders()
    return make_windows_from_text(text, window_size, vowel)
