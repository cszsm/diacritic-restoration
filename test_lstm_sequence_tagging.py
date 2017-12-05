import os.path
from keras.models import load_model
from src.train.lstm_sequence_tagging import _calculate_fscores

import numpy as np

path = os.path.join('models', 'bidirectional_lstm_sequence_tagging',
                    'trained.model')
model = load_model(path)

data = np.load(
    os.path.join('res', 'prepared', 'lstm_sequence_tagging', 'prepared.npz'))

_calculate_fscores(model, data['x'], data['y'])