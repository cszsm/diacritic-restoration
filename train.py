import argparse

from src.train import training_framework

parser = argparse.ArgumentParser()
# parser.add_argument('model', help='can be \'feedforward\' or \'lstm_baseline\'')
parser.add_argument('--units', type=int)
parser.add_argument('--window_size', type=int)

args = parser.parse_args()

framework = training_framework.Framework('lstm_baseline')

if args.units is None and args.window_size is None:
    framework.run()
elif args.units is not None and args.window_size is not None:
    params_list = []
    params = {}
    params['units'] = args.units
    params['window_size'] = args.window_size
    params_list.append(params)

    framework.run(params_list)
else:
    print('ERROR: Either both parameters must be given or none of them')
