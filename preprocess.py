import argparse

from src.preprocess import preprocessor_framework
from src.train import lstm_baseline_network

parser = argparse.ArgumentParser(description='Script for preprocessing the corpus for the baseline lstm neural network.')
# parser.add_argument('preprocessor', help='can be \'feedforward\' or \'lstm_baseline\'')
parser.add_argument('count', type=int)
parser.add_argument('--window_size', type=int)
# parser.add_argument('--vowel', help='if not given, the corpus will be processed for all vowels')

args = parser.parse_args()

params_list = lstm_baseline_network.Network.get_exhaustive_parameters()

framework = preprocessor_framework.PreprocessorFramework('lstm_baseline')

if args.window_size:
    framework.process(args.count, args.window_size)
else:
    for params in params_list:
        framework.process(args.count, params['window_size'])
