from src import accentizer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('text')

args = parser.parse_args()
text = args.text

accentized = accentizer.accentize_with_lstm_sequence_tagging(text)

print(accentized)
