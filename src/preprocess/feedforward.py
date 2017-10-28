'''
Preprocesses data for the feedforward network.
'''

import numpy as np
from collections import deque

from sklearn.feature_extraction import DictVectorizer

from src.preprocess.common import transform_accent, normalize_character, deaccentize

# TODO: rename accent_table
vowel_table = {
    'a': ['a', 'á'],
    'e': ['e', 'é'],
    'i': ['i', 'í'],
    'o': ['o', 'ó', 'ö', 'ő'],
    'u': ['u', 'ú', 'ü', 'ű']
}


def process_for_train(count, window_size, vowel, words):

    preprocessor = _Preprocessor(count, window_size, vowel)
    windows, accents = preprocessor.process(words)

    return windows, accents


def process_for_accentize(window_size, words):

    vectorizer = DictVectorizer()
    vectorizer.fit(_generate_windows(window_size))

    windows = {}

    for vowel in vowel_table.keys():
        preprocessor = _Preprocessor(-1, window_size, vowel)
        windows[vowel], _ = preprocessor.process(words)

    return windows


class _Preprocessor:
    '''Preprocessor for the feedforward network.
    If count is set to -1, it will not count the accents and will process all of the given words.'''

    def __init__(self, count, window_size, vowel):
        self.count = count
        self.window_size = window_size
        self.vowel = vowel

        self.accent_count = {}
        for accent in vowel_table[vowel]:
            self.accent_count[accent] = 0

    def process(self, words):

        processed_windows = []
        processed_accents = []

        vectorizer = DictVectorizer()
        vectorizer.fit(_generate_windows(self.window_size))

        word_counter = _WordCounter(self.vowel)

        for word in words:

            # If the current word does not contain any of the vowel's accent, we skip it
            skip = True
            for accent in vowel_table[self.vowel]:
                if accent in word:
                    skip = False
                    break
            if skip:
                continue

            windows, accent_classes = self._process_word(word.lower())

            # TODO: why would it be
            if len(windows) == 0:
                continue

            # transformed_windows = _transform_windows(vectorizer, windows)

            transformed_windows = vectorizer.transform(windows)
            if processed_windows == []:
                processed_windows = transformed_windows.toarray()
            else:
                for transformed_window in transformed_windows:
                    processed_windows = np.concatenate(
                        processed_windows, transformed_window.toarray())
            processed_accents += accent_classes

            word_counter.increase(self.accent_count)

            return processed_windows, processed_accents

    def _process_word(self, word):
        windows = []
        accents = []

        sliding_window = deque((), self.window_size * 2 + 1)
        for i in range(sliding_window.maxlen):
            sliding_window.append('_')
            word += '_'

        for character in word:
            sliding_window.append(character)

            current_character = sliding_window[self.window_size]

            character_is_vowel = current_character in vowel_table[self.vowel]

            if character_is_vowel:

                not_enough_accent = self.accent_count[current_character] < self.count
                do_not_count = self.count == -1

                if not_enough_accent or do_not_count:

                    self.accent_count[current_character] += 1

                    transformed_window = self._transform_window(
                        sliding_window.copy())
                    transformed_accent = transform_accent(current_character)

                    windows.append(transformed_window)
                    accents.append(transformed_accent)

        return windows, accents

    def _transform_window(self, window):
        transformed_window = {}

        for i in range(-self.window_size, self.window_size + 1):
            character = window.popleft()
            deaccentized = deaccentize(character)
            normalized = normalize_character(deaccentized)
            transformed_window[i] = normalized

        del transformed_window[0]

        return transformed_window


def _generate_windows(window_size):
    '''Generates template windows to fit the vectorizer with.
    Each character in the alphabet occurs at each index in a window.'''

    windows = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz 0_*'
    alphabet_size = len(alphabet)

    for i in range(alphabet_size):

        # Getting a slice from the alphabet as a string
        end_of_slice = i + window_size * 2
        if end_of_slice <= alphabet_size:
            alphabet_slice = alphabet[i:end_of_slice]
        else:
            alphabet_slice = alphabet[i:alphabet_size]
            alphabet_slice += alphabet[0:end_of_slice - alphabet_size]

        # Creating a window from the slice
        new_window = {}
        for j in range(window_size):
            new_window[-1 * (j + 1)] = alphabet_slice[window_size - 1 - j]
            new_window[j + 1] = alphabet_slice[window_size + j]

        windows.append(new_window)

    return windows


class _WordCounter:
    def __init__(self, vowel):
        self.count = 0
        self.vowel = vowel

    def increase(self, accent_count):
        '''Increases the counter. Logs progress after every 100th word'''

        self.count += 1

        if self.count % 100 == 0:
            print('word: ' + str(self.count))
            for accent in vowel_table[self.vowel]:
                print('\t' + accent + ': ' + str(accent_count[accent]))
