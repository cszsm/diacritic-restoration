from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import numpy as np

ALPHABET = "aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz"

LABEL_ENC = LabelEncoder()
ONEHOT_ENC = OneHotEncoder(sparse=False)


def deaccentize(text):
    '''Removes diacritics from hungarian text'''

    text = text.replace('á', 'a')
    text = text.replace('é', 'e')
    text = text.replace('í', 'i')
    text = text.replace('ó', 'o')
    text = text.replace('ö', 'o')
    text = text.replace('ő', 'o')
    text = text.replace('ú', 'u')
    text = text.replace('ü', 'u')
    text = text.replace('ű', 'u')

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
    '''Reduces the number of different characters to 39'''
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
    alphabet = 'abcdefghijklmnopqrstuvwxyz 0_*'
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


def tag_character(character):
    '''Used for sequence tagging'''

    if character in 'aeiou':
        return [0, 1, 0, 0, 0]
    if character in 'áéíóú':
        return [0, 0, 1, 0, 0]
    if character in 'öü':
        return [0, 0, 0, 1, 0]
    if character in 'őű':
        return [0, 0, 0, 0, 1]
    return [1, 0, 0, 0, 0]


def accentize_by_tag(character, tag):
    '''Used for sequence tagging'''

    # TODO
    accents_1 = {'a': 'á', 'e': 'é', 'i': 'í', 'o': 'ó', 'u': 'ú'}
    accents_2 = {'o': 'ö', 'u': 'ü'}
    accents_3 = {'o': 'ő', 'u': 'ű'}

    # TODO argmax?
    if tag is 2:
        return accents_1[character]
    if tag is 3:
        return accents_2[character]
    if tag is 4:
        return accents_3[character]
    return character
