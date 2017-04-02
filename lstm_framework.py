from sklearn.model_selection import train_test_split
import numpy as np

import lstm_baseline_network as network
import lstm_baseline_preprocessor as preprocessor

class Network:

    CURRENT_VOWEL = 'a'

    def run(self):
        model = network.get_model()

        prepared_data = np.load("prepared_" + self.CURRENT_VOWEL + ".npz")

        train_x, test_x, train_y, test_y = train_test_split(prepared_data["x"], prepared_data["y"], test_size=0.2)

        model.fit(train_x, train_y, batch_size=32, epochs=100)
        score = model.evaluate(test_x, test_y, batch_size=32)

        print(score)

        x, y = preprocessor.make_windows(["áradat"], 1, 'a')
        print(x)
        print(y)
        print(model.predict(x))
        