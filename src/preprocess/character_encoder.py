from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ENGLISH_ALPHABET = list('abcdefghijklmnopqrstuvwxyz 0_*')
# HUNGARIAN_ALPHABET = list('aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz 0_*')


class _Encoder:

    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    alphabet = []

    def __init__(self):

        # if alphabet is 'english':
        #     self.label_encoder.fit(ENGLISH_ALPHABET)
        # elif alphabet is 'hungarian':
        #     self.label_encoder.fit(HUNGARIAN_ALPHABET)

        self.label_encoder.fit(self.alphabet)

        labels = self.label_encoder.transform(self.alphabet)
        self.onehot_encoder.fit(labels.reshape(-1, 1))

    def transform(self, character):

        label = self.label_encoder.transform([character])[0]
        onehot = self.onehot_encoder.transform(label)[0]

        return onehot


class EnglishEncoder(_Encoder):

    alphabet = list('abcdefghijklmnopqrstuvwxyz 0_*')


class HungarianEncoder(_Encoder):

    alphabet = list('aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz 0_*')
