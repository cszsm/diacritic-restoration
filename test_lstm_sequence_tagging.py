import os.path
from keras.models import load_model
from src.train.lstm_sequence_tagging import _calculate_fscores

import numpy as np
from src.preprocess.lstm_sequence_tagging import process

path = os.path.join('models', 'bidirectional_lstm_sequence_tagging',
                    'trained.model')
model = load_model(path)

with open(os.path.join('res', 'sentences'), encoding='utf8') as f:
    test_x, test_y = [], []
    for i in range(200):
        x, y = process(f.readline().rstrip('\n'))
        test_x.append(x[0])
        test_y.append(y[0])

    _calculate_fscores(model, test_x, test_y)