from src.train.training_framework import Framework
from src.train.feedforward_network import Network

framework = Framework('feedforward')

framework.run(Network.get_exhaustive_parameters())