from sklearn.model_selection import train_test_split
import numpy as np

import lstm_baseline_network as network
import logger as logger

import datetime
import os.path
import argparse

LATIN_VOWELS = ['a', 'e', 'i', 'o', 'u']

class Framework:

    train_x = {}
    train_y = {}
    test_x = {}
    test_y = {}

    def run_network(self, params, vowel, id):
        net = network.Network(params, vowel, self.logger)

        net.run(self.train_x[vowel], self.test_x[vowel], self.train_y[vowel], self.test_y[vowel])

        self.model = net.get_model()
        self.save_model(vowel, id)
        
    def run(self, count):
        self.load_prepared_data()

        self.filename = self.create_filename()
        self.logger = logger.Logger(self.filename)

        for i in range(count):
            params = network.Network.get_random_parameters()
            self.logger.log('\nid: model_' + str(i))
            self.logger.log('\n\nunits: ' + str(params['units']))
            os.makedirs(os.path.join('../models', self.filename + '_' + str(i)))
            for vowel in LATIN_VOWELS:
                self.run_network(params, vowel, str(i))

    def create_filename(self):
        date = str(datetime.datetime.today().date())
        filename = date

        i = 1
        while(os.path.isfile(os.path.join('../logs', filename))):
            filename = date + "-" + str(i)
            i += 1
            print(filename)

        return filename

    def load_prepared_data(self):
        for vowel in LATIN_VOWELS:
            prepared_data = np.load(os.path.join("../res", "prepared_" + vowel + ".npz"))
            self.train_x[vowel], self.test_x[vowel], self.train_y[vowel], self.test_y[vowel] = train_test_split(prepared_data['x'], prepared_data['y'], test_size=0.2)


    def save_model(self, vowel, id):
        self.model.save(os.path.join('../models', self.filename + '_' + id, vowel + '.model'))


parser = argparse.ArgumentParser()
parser.add_argument("count")

args = parser.parse_args()
count = int(args.count)

framework = Framework()
framework.run(count)
