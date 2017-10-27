'''Encoders for transforming characters to onehots'''
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class _Encoder:
    def __init__(self, alphabet):

        self.label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse=False)

        self.label_encoder.fit(alphabet)

        labels = self.label_encoder.transform(alphabet)
        self.onehot_encoder.fit(labels.reshape(-1, 1))

    def transform(self, character):
        '''Transforms a character to onehot'''

        label = self.label_encoder.transform([character])[0]
        onehot = self.onehot_encoder.transform(label)[0]

        return onehot

    def inverse_transform(self, onehot):
        '''Transforms onehot back to original character'''

        label = np.array(onehot).argmax(axis=0)
        character = self.label_encoder.inverse_transform(label)

        return character


class EnglishEncoder(_Encoder):
    '''Encoder for letters without diacritics'''

    def __init__(self):
        super().__init__(list('abcdefghijklmnopqrstuvwxyz 0_*'))


class HungarianEncoder(_Encoder):
    '''Encoder for letters with hungarian diacritics'''

    def __init__(self):
        super().__init__(list('aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz 0_*'))
