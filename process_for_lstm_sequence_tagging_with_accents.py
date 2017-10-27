import argparse
from src.preprocess.preprocessor_framework import PreprocessorFramework

parser = argparse.ArgumentParser()
parser.add_argument('count', type=int)

args = parser.parse_args()

framework = PreprocessorFramework('lstm_sequence_tagging_with_accents')

# TODO
framework.process(args.count, 0)
