from sklearn.model_selection import train_test_split
import numpy as np

from src.train import feedforward_network
from src.train import lstm_baseline_network
from src.train.lstm_sequence_tagging import Network as LstmSequenceTaggingNetwork

from keras.preprocessing.sequence import pad_sequences

from src.logger import Logger

import datetime
import os.path
# import argparse

import sys

LATIN_VOWELS = ['a', 'e', 'i', 'o', 'u']

# LATIN_VOWELS = ['u']


class Framework:
    def __init__(self, model):
        self.model_name = model

        # TODO network
        if self.model_name == 'feedforward':
            self.network = feedforward_network
        elif self.model_name == 'lstm_baseline':
            self.network = lstm_baseline_network
        elif self.model_name == 'lstm_sequence_tagging':
            self.network = LstmSequenceTaggingNetwork

        # self.log_path = '../logs/' + self.model_name
        self.model_path = os.path.join('models', self.model_name)
        self.res_path = 'res'

        self.prepared_data = {}
        self.prepared_data['train_x'] = {}
        self.prepared_data['train_y'] = {}
        self.prepared_data['valid_x'] = {}
        self.prepared_data['valid_y'] = {}
        self.prepared_data['test_x'] = {}
        self.prepared_data['test_y'] = {}

        # self.filename = self.create_filename()
        # self.logger = logger.Logger(self.model_name)

    def _run_network(self, params, vowel):
        """Runs a network with the given vowel and parameters."""
        net = {}

        # TODO
        if self.model_name == 'feedforward':
            feedforward_logger = Logger(self.model_name + '_f' + str(
                params['first']) + '_s' + str(params['second']) + '_t' + str(
                    params['third']) + '_' + vowel)
            self.network.Network.log_parameters(feedforward_logger, params)

            net = feedforward_network.Network(params, vowel,
                                              feedforward_logger)
            net.run(self.prepared_data['train_x'][vowel],
                    self.prepared_data['train_y'][vowel],
                    self.prepared_data['valid_x'][vowel],
                    self.prepared_data['valid_y'][vowel],
                    self.prepared_data['test_x'][vowel],
                    self.prepared_data['test_y'][vowel], params['hidden'])
            # self.save_model(net.get_model(), vowel, id)
            net.save_model(
                os.path.join(self.model_path,
                             str(params['first']),
                             str(params['second']),
                             str(params['third']), vowel + '.model'))

        elif self.model_name == 'lstm_baseline':
            lstm_logger = Logger(self.model_name + '_u' +
                                 str(params['units']) + '_w' + str(
                                     params['window_size']) + '_' + vowel)
            self.network.Network.log_parameters(lstm_logger, params)

            net = lstm_baseline_network.Network(params, vowel, lstm_logger)
            net.run(self.prepared_data['train_x'][vowel],
                    self.prepared_data['test_x'][vowel],
                    self.prepared_data['train_y'][vowel],
                    self.prepared_data['test_y'][vowel])
            # self.save_model(net.get_model(), vowel, id)
            net.get_model().save(
                os.path.join(self.model_path,
                             str(params['units']),
                             str(params['window_size']), vowel + '.model'))

        elif self.model_name == 'lstm_sequence_tagging' or self.model_name == 'bidirectional_lstm_sequence_tagging':
            logger = Logger(self.model_name)
            # LstmSequenceTaggingNetwork.log_parameters(logger)

            if self.model_name == 'lstm_sequence_tagging':
                net = LstmSequenceTaggingNetwork(logger, False)
            else:
                net = LstmSequenceTaggingNetwork(logger, True)

            # net.train(self.prepared_data['x'][:20000],
            #           self.prepared_data['y'][:20000])

            net.train()

            net.get_model().save(
                os.path.join(self.model_path, 'trained.model'))

        # self.logger.log('\n\n')

    def run(self, params_list=None):
        # TODO
        if self.model_name == 'lstm_sequence_tagging' or self.model_name == 'bidirectional_lstm_sequence_tagging':
            self._load_prepared_data(0)
            os.makedirs(self.model_path, exist_ok=True)

            # TODO
            self._run_network([], 0)

            return

        if params_list == None:
            """Runs networks with exhaustive parameters, each with all vowels."""
            params_list = self.network.Network.get_exhaustive_parameters()

        for params in params_list:
            if self.model_name == 'feedforward':
                self.prepared_data = self._load_prepared_data('4')
                os.makedirs(
                    os.path.join(self.model_path,
                                 str(params['first']),
                                 str(params['second']), str(params['third'])),
                    exist_ok=True)
            elif self.model_name == 'lstm_baseline':
                self.prepared_data = self._load_prepared_data(
                    params['window_size'])
                os.makedirs(
                    os.path.join(self.model_path,
                                 str(params['units']),
                                 str(params['window_size'])),
                    exist_ok=True)

            for vowel in LATIN_VOWELS:
                self._run_network(params, vowel)

    def create_filename(self):
        """Creates a filename for logs and saved models."""
        date = str(datetime.datetime.today().date())
        filename = date

        i = 1
        # print('ezitt: ' + os.path.join(self.log_path, filename))
        while (os.path.isfile(os.path.join(self.log_path, filename))):
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

    def _load_prepared_data(self, window_size):
        """Loads the prepared data for training, validating and testing."""
        prepared = {}
        prepared['train_x'] = {}
        prepared['train_y'] = {}
        #TODO
        prepared['valid_x'] = {}
        prepared['valid_y'] = {}
        prepared['test_x'] = {}
        prepared['test_y'] = {}

        if self.model_name == 'feedforward':
            for vowel in LATIN_VOWELS:
                prepared_data = np.load(
                    os.path.join(self.res_path, 'prepared', self.model_name,
                                 str(window_size), vowel + '.npz'))
                prepared['train_x'][vowel], valid_test_x, prepared['train_y'][
                    vowel], valid_test_y = train_test_split(
                        prepared_data['x'], prepared_data['y'], test_size=0.4)
                prepared['test_x'][vowel], prepared['valid_x'][
                    vowel], prepared['test_y'][vowel], prepared['valid_y'][
                        vowel] = train_test_split(
                            valid_test_x, valid_test_y, test_size=0.5)

                print('____________________________________')
                print(len(prepared['train_x'][vowel]))
        elif self.model_name == 'lstm_baseline':
            for vowel in LATIN_VOWELS:
                prepared_data = np.load(
                    os.path.join(self.res_path, 'prepared', self.model_name,
                                 str(window_size), vowel + '.npz'))
                prepared['train_x'][vowel], prepared['test_x'][
                    vowel], prepared['train_y'][vowel], prepared['test_y'][
                        vowel] = train_test_split(
                            prepared_data['x'],
                            prepared_data['y'],
                            test_size=0.2)
        elif self.model_name == 'lstm_sequence_tagging' or self.model_name == 'bidirectional_lstm_sequence_tagging':
            prepared_data = np.load(
                os.path.join(self.res_path, 'prepared',
                             'lstm_sequence_tagging', 'prepared.npz'))

        # TODO
        self.prepared_data = prepared_data
        return prepared

    def save_model(self, model, vowel, id):
        """Saves the model."""
        model.save(
            os.path.join(self.model_path, self.filename + '_' + id,
                         vowel + '.model'))


# parser = argparse.ArgumentParser()
# parser.add_argument('model', help='can be \'feedforward\' or \'lstm_baseline\'')
# # parser.add_argument('count', type=int)

# args = parser.parse_args()

# framework = Framework(args.model)
# framework.run()
