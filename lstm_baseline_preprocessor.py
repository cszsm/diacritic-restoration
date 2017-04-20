"""Preprocess data for the baseline lstm network"""
from collections import deque
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
import numpy as np

ALPHABET = "aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz"
VOWELS = "aáeéiíoóöőuúüű"
VOWEL_TABLE = {"a": ["á"], "e": ["é"], "i": ["í"],
               "o": ["ó", "ö", "ő"], "u": ["ú", "ü", "ű"]}
VECTORIZER = DictVectorizer()

LABEL_ENC = LabelEncoder()
ONEHOT_ENC = OneHotEncoder(sparse=False)


def deaccentize(text):
    text = text.replace("á", "a")
    text = text.replace("é", "e")
    text = text.replace("í", "i")
    text = text.replace("ó", "o")
    text = text.replace("ö", "o")
    text = text.replace("ő", "o")
    text = text.replace("ú", "u")
    text = text.replace("ü", "u")
    text = text.replace("ű", "u")

    return text

def deaccentize_list(character_list):
    deaccentized_list = []

    for character in character_list:
        deaccentized_list += deaccentize(character)

    return deaccentized_list

def ispunct(c):
    punctuations = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for char in punctuations:
        if c == char:
            return True
    return False


def isalpha(c):
    for char in ALPHABET:
        if c == char:
            return True
    return False


def normalize_character(c):
    """reduces the number of different characters to 39"""
    if c.isspace():
        return ' '
    if c.isdigit():
        return '0'
    if ispunct(c):
        return '_'
    if isalpha(c):
        return c
    return '*'


def normalize_list(character_list):
    normalized_list = []

    for character in character_list:
        normalized_list.append(normalize_character(deaccentize(character)))

    return normalized_list


def normalize_text(text):
    normalized_text = ""

    for character in text:
        normalized_text += normalize_character(character)

    return normalized_text


def fit_encoders():
    alphabet = "abcdefghijklmnopqrstuvwxyz 0_*"
    alphabet_list = list(alphabet)
    
    LABEL_ENC.fit(alphabet_list)
    label_list = LABEL_ENC.transform(alphabet_list)
    ONEHOT_ENC.fit(label_list.reshape(-1, 1))


def transform(character):
    character_label = LABEL_ENC.transform([character])
    return ONEHOT_ENC.transform(character_label[0])[0]


def transform_list(character_list):
    transformed_list = []

    for character in character_list:
        transformed_character = transform(character)
        transformed_list.append(transformed_character)

    return transformed_list


def transform_accent(vowel):
    if vowel in "aei":
        return [1, 0]
    if vowel in "áéí":
        return [0, 1]
    if vowel in "ou":
        return [1, 0, 0, 0]
    if vowel in "óú":
        return [0, 1, 0, 0]
    if vowel in "öü":
        return [0, 0, 1, 0]
    if vowel in "őű":
        return [0, 0, 0, 1]


def pad_word(word, window_size):
    if window_size > 0:
        return '_' + pad_word(word, window_size - 1) + '_'
    return word


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
            normalized_list = list(deaccentize_list(list(sliding_window)))
            transformed_list = transform_list(normalized_list)
            transformed_accents = sliding_window[window_size]

            windows.append(transformed_list)
            accents.append(transform_accent(transformed_accents))

    return windows, accents


def make_windows_from_text(text, window_size, vowel):
    windows = []
    accents = []

    for word in text:
        normalized_word = normalize_text(word)
        padded_word = pad_word(normalized_word.lower(), window_size)

        new_windows, new_accents = make_windows_from_word(
            padded_word, window_size, vowel)

        windows += new_windows
        accents += new_accents

    return windows, accents

def make_windows(text, window_size, vowel):
    fit_encoders()
    return make_windows_from_text(text, window_size, vowel)
