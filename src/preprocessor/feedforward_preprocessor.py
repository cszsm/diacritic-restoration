from collections import deque
from sklearn.feature_extraction import DictVectorizer

import numpy as np

import preprocessor_common as common

vowel_table = {'a': ['a', 'á'], 'e': ['e', 'é'], 'i': ['i', 'í'], 'o': ['o', 'ó', 'ö', 'ő'], 'u': ['u', 'ú', 'ü', 'ű']}
vectorizer = DictVectorizer()

class FeedforwardPreprocessor:

    accent_counter = {}

    def __init__(self, count, window_size, vowel):
        self.count = count
        self.window_size = window_size
        self.vowel = vowel

    def create_row(self, window):
        row = {}

        for i in range(-self.window_size, self.window_size + 1):
            row[i] = common.normalize_character(common.deaccentize(window.popleft()))

        del row[0]

        return row

    # returns
    # x_e: list of windows with the given window_size
    # y_e: one-hot encoded values
    def make_windows_from_word(self, word):
        x_e = []
        y_e = []
        lower_text = word.lower()

        window = deque((), self.window_size * 2 + 1)
        for i in range(window.maxlen):
            window.append("_")
            lower_text += "_"

        for character in lower_text:
            window.append(character)
            
            if (window[self.window_size] in vowel_table[self.vowel]) and (self.accent_counter[window[self.window_size]] < self.count):

                self.accent_counter[window[self.window_size]] += 1

                x_e.append(self.create_row(window.copy()))
                y_e.append(common.transform_accent(window[self.window_size]))

        return x_e, y_e

    def make_windows_from_text(self, text):
        print("")
        x_e = []
        y_e = []

        for accent in vowel_table[self.vowel]:
            self.accent_counter[accent] = 0
        
        # count = 0
        
        word_counter = 0
        for word in text:
            word_counter += 1

            skip = True
            # if self.vowel in word:
            #     skip = False
            for c in vowel_table[self.vowel]:
                if c in word:
                    skip = False
            if skip:
                continue
                
            x, y = self.make_windows_from_word(word)
            # print(x)
            # print(y)

            if len(x) == 0:
                continue

            if x_e == []:
                x_e = vectorizer.transform(x).toarray()
            else:
                # print(x)
                tmp = vectorizer.transform(x)
                for tx in tmp:
                    #x_e += tx.toarray()
                    x_e = np.concatenate((x_e, tx.toarray()))
            y_e += y
            # count += 1
            # if count % 500 == 0:
            #     print("|", end="")

            if word_counter % 100 == 0:
                print('word: ' + str(word_counter))
                for accent in vowel_table[self.vowel]:
                    print("\t" + accent + ": " + str(self.accent_counter[accent]))
            
        return x_e, y_e


    # generates template windows for the alphabet
    def generate_windows(self):
        windows = []
        alphabet = "abcdefghijklmnopqrstuvwxyz 0_*"
        alphabet_size = len(alphabet)

        for i in range(alphabet_size):
            new_window = {}

            end_of_slice = i + self.window_size * 2
            if end_of_slice <= alphabet_size:
                alphabet_slice = alphabet[i:end_of_slice]
            else:
                alphabet_slice = alphabet[i:alphabet_size]
                alphabet_slice += alphabet[0:end_of_slice - alphabet_size]

            for j in range(self.window_size):
                new_window[-1 * (j + 1)] = alphabet_slice[self.window_size - 1 - j]
                new_window[j + 1] = alphabet_slice[self.window_size + j]

            windows.append(new_window)

        return windows

    def make_windows(self, text):
        vectorizer.fit(self.generate_windows())
        return self.make_windows_from_text(text)