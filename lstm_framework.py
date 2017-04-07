from sklearn.model_selection import train_test_split
import numpy as np

import lstm_baseline_network as network
import lstm_baseline_preprocessor as preprocessor

class Framework:

    CURRENT_VOWEL = 'a'

    def run(self, train_x, test_x, train_y, test_y):
        net = network.Network()

        net.run(train_x, test_x, train_y, test_y)

        model = net.get_model()

        x, y = preprocessor.make_windows(["áradat"], 1, 'a')
        print(x)
        print(y)
        print(model.predict(x))
        
    def run_more(self, count):
        prepared_data = np.load("prepared_" + self.CURRENT_VOWEL + ".npz")

        train_x, test_x, train_y, test_y = train_test_split(prepared_data["x"], prepared_data["y"], test_size=0.2)

        for i in range(count):
            self.run(train_x, test_x, train_y, test_y)


Framework().run_more(3)