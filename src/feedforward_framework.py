import feedforward_network as network
import logger

class Framework:

    log_path = '../logs/feedforward'
    model_path = '../models'
    res_path = '../res'

    train_x = {}
    train_y = {}
    valid_x = {}
    valid_y = {}
    test_x = {}
    test_y = {}

    def run_network(self, params, vowel, id):
        """Runs a network with the given vowel and parameters."""
        net = network.Network(params, vowel, self.logger)

        net.run(self.train_x[vowel], self.train_y[vowel], self.valid_x[vowel], self.valid_y[vowel], self.test_x[vowel], self.test_y[vowel])

        self.model = net.get_model()
        self.save_model(vowel, id)
        
    
    def run(self, count):
        """Runs networks with random parameters, each with all vowels."""
        self.load_prepared_data()

        self.filename = self.create_filename()
        self.logger = logger.Logger(self.filename)

        for i in range(count):
            params = network.Network.get_random_parameters()
            self.logger.log



    def create_filename(self):
        """Creates a filename for logs and saved models."""
        pass

    def load_prepared_data(self):
        """Loads the prepared data for training, validating and testing."""
        pass

    def save_model(self, vowel, id):
        """Saves the model."""
        pass