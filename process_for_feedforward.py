import argparse
from src.preprocess.preprocessor_framework import process_for_feedforward

parser = argparse.ArgumentParser()
parser.add_argument('count', type=int)
parser.add_argument('window_size', type=int)

args = parser.parse_args()

process_for_feedforward(args.count, args.window_size)
