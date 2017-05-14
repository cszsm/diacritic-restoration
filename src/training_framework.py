from sklearn.model_selection import train_test_split
import numpy as np

import feedforward_network
import lstm_baseline_network
import logger

import datetime
import os.path
import argparse

# LATIN_VOWELS = ['a', 'e', 'i', 'o', 'u']
LATIN_VOWELS = ['u']

class Framework:

    def __init__(self, model):
        self.model_name = model

        if self.model_name == 'feedforward':
            self.network = feedforward_network
        elif self.model_name == 'lstm_baseline':
            self.network = lstm_baseline_network

        # self.log_path = '../logs/' + self.model_name
        self.model_path = '../models/' + self.model_name
        self.res_path = '../res'

        self.prepared_data = {}
        self.prepared_data['train_x'] = {}
        self.prepared_data['train_y'] = {}
        self.prepared_data['valid_x'] = {}
        self.prepared_data['valid_y'] = {}
        self.prepared_data['test_x'] = {}
        self.prepared_data['test_y'] = {}

        # self.filename = self.create_filename()
        self.logger = logger.Logger(self.model_name)

    def run_network(self, params, vowel):
        """Runs a network with the given vowel and parameters."""
        net = {}

        # TODO
        if self.model_name == 'feedforward':
            pass
            # TODO
            # net = feedforward_network.Network(params, vowel, self.logger)
            # net.run(self.prepared_data['train_x'][vowel], self.prepared_data['train_y'][vowel], self.prepared_data['valid_x'][vowel], self.prepared_data['valid_y'][vowel], self.prepared_data['test_x'][vowel], self.prepared_data['test_y'][vowel])
            # # self.save_model(net.get_model(), vowel, id)
            # net.save_model(os.path.join(self.model_path, params['units'], params['window_size'], vowel + '.model'))
        elif self.model_name == 'lstm_baseline':
            net = lstm_baseline_network.Network(params, vowel, self.logger)
            net.run(self.prepared_data['train_x'][vowel], self.prepared_data['test_x'][vowel], self.prepared_data['train_y'][vowel], self.prepared_data['test_y'][vowel])
            # self.save_model(net.get_model(), vowel, id)
            net.get_model().save(os.path.join(self.model_path, str(params['units']), str(params['window_size']), vowel + '.model'))

        self.logger.log('\n\n')



    # def run(self, count):
    #     """Runs networks with random parameters, each with all vowels."""
    #     self.load_prepared_data()

    #     for i in range(count):
    #         params = self.network.Network.get_random_parameters()
    #         self.logger.log('\nid: session_' + str(i))
    #         self.network.Network.log_parameters(self.logger, params)
    #         os.makedirs(os.path.join(self.model_path, self.filename + '_' + str(i)))
    #         for vowel in LATIN_VOWELS:
    #             self.run_network(params, vowel, str(i))


    def run(self):
        """Runs networks with exhaustive parameters, each with all vowels."""
        params_list = self.network.Network.get_exhaustive_parameters()

        for params in params_list:
            self.prepared_data = self.load_prepared_data(params['window_size'])
            self.network.Network.log_parameters(self.logger, params)
            os.makedirs(os.path.join(self.model_path, str(params['units']), str(params['window_size'])), exist_ok=True)
            for vowel in LATIN_VOWELS:
                self.run_network(params, vowel)


    def create_filename(self):
        """Creates a filename for logs and saved models."""
        date = str(datetime.datetime.today().date())
        filename = date

        i = 1
        print('ezitt: ' + os.path.join(self.log_path, filename))
        while(os.path.isfile(os.path.join(self.log_path, filename))):
            filename = date + "-" + str(i)
            i += 1
            print(filename)

        return filename

    # def load_prepared_data(self):
    #     """Loads the prepared data for training, validating and testing."""
    #     if self.model_name == 'feedforward':
    #         for vowel in LATIN_VOWELS:
    #             prepared_data = np.load(os.path.join(self.res_path, self.model_name + "_prepared_" + vowel + ".npz"))
    #             self.prepared_data['train_x'][vowel], valid_test_x, self.prepared_data['train_y'][vowel], valid_test_y = train_test_split(prepared_data['x'], prepared_data['y'], test_size=0.4)
    #             self.prepared_data['test_x'][vowel], self.prepared_data['valid_x'][vowel], self.prepared_data['test_y'][vowel], self.prepared_data['valid_y'][vowel] = train_test_split(prepared_data['x'], prepared_data['y'], test_size=0.5)
    #     elif self.model_name == 'lstm_baseline':
    #         for vowel in LATIN_VOWELS:
    #             prepared_data = np.load(os.path.join(self.res_path, self.model_name + "_prepared_" + vowel + ".npz"))
    #             self.prepared_data['train_x'][vowel], self.prepared_data['test_x'][vowel], self.prepared_data['train_y'][vowel], self.prepared_data['test_y'][vowel] = train_test_split(prepared_data['x'], prepared_data['y'], test_size=0.2)


    def load_prepared_data(self, window_size):
        """Loads the prepared data for training, validating and testing."""
        prepared = {}
        prepared['train_x'] = {}
        prepared['train_y'] = {}
        # prepared['valid_x'] = {}
        # prepared['valid_y'] = {}
        prepared['test_x'] = {}
        prepared['test_y'] = {}

        if self.model_name == 'feedforward':
            pass
            # TODO
            # for vowel in LATIN_VOWELS:
            #     prepared_data = np.load(os.path.join(self.res_path, 'prepared', self.model_name, str(window_size), vowel + ".npz"))
            #     self.prepared_data['train_x'][vowel], valid_test_x, self.prepared_data['train_y'][vowel], valid_test_y = train_test_split(prepared_data['x'], prepared_data['y'], test_size=0.4)
            #     self.prepared_data['test_x'][vowel], self.prepared_data['valid_x'][vowel], self.prepared_data['test_y'][vowel], self.prepared_data['valid_y'][vowel] = train_test_split(prepared_data['x'], prepared_data['y'], test_size=0.5)
        elif self.model_name == 'lstm_baseline':
            for vowel in LATIN_VOWELS:
                prepared_data = np.load(os.path.join(self.res_path, 'prepared', self.model_name, str(window_size), vowel + ".npz"))
                prepared['train_x'][vowel], prepared['test_x'][vowel], prepared['train_y'][vowel], prepared['test_y'][vowel] = train_test_split(prepared_data['x'], prepared_data['y'], test_size=0.2)
                
        return prepared


    def save_model(self, model, vowel, id):
        """Saves the model."""
        model.save(os.path.join(self.model_path, self.filename + '_' + id, vowel + '.model'))

parser = argparse.ArgumentParser()
parser.add_argument('model', help='can be \'feedforward\' or \'lstm_baseline\'')
# parser.add_argument('count', type=int)

args = parser.parse_args()

framework = Framework(args.model)
framework.run()
