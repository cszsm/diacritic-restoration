import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src import accentizer

# tf.logging.set_verbosity(tf.logging.ERROR)

text = input()
accentized = accentizer.accentize_with_lstm_sequence_tagging(text)

print(accentized)