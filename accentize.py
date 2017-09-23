import argparse

from src import accentizer

parser = argparse.ArgumentParser()
# parser.add_argument('network_type', help='can be \'feedforward\' or \'lstm_baseline\'')
# parser.add_argument('--model_id')
parser.add_argument('units', type=int)
parser.add_argument('window_size', type=int)
parser.add_argument('text')

args = parser.parse_args()

accentizer.accentize(args.text, 'lstm_baseline', args.units, args.window_size)