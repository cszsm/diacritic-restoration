from src.preprocess.corpus_reader import CorpusReader, read_sentences

from src.preprocess.lstm_baseline_preprocessor import LstmBaselinePreprocessor
from src.preprocess.feedforward_preprocessor import FeedforwardPreprocessor
from src.preprocess.lstm_sequence_tagging_with_accents import process_for_train as process_for_lstm_sequence_tagging_with_accents
from src.preprocess.lstm_sequence_tagging import process as process_for_lstm_sequence_tagging

import numpy as np
import os.path

VOWEL_TABLE = {
    'a': ['a', 'á'],
    'e': ['e', 'é'],
    'i': ['i', 'í'],
    'o': ['o', 'ó', 'ö', 'ő'],
    'u': ['u', 'ú', 'ü', 'ű']
}
RESOURCE_DIRECTORY = 'res'


class PreprocessorFramework:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def process(self, count, window_size, vowel=None):

        if self.preprocessor == 'lstm_sequence_tagging':
            parent_path = self.create_parent_path()

            sentences = read_sentences(count)

            # TODO rename
            characters, tags = process_for_lstm_sequence_tagging(sentences)

            path = os.path.join(parent_path, 'prepared')
            np.savez(path, x=characters, y=tags)

            file = open(os.path.join(parent_path, '_count.txt'), 'w')
            file.write(str(count))

            #TODO
            return

        if (self.preprocessor != 'lstm_sequence_tagging_with_accents'):
            if vowel:
                words = CorpusReader.read_words(vowel, count)
                px, py = self.create_preprocessor(count, window_size,
                                                  vowel).make_windows(words)
                np.savez(self.create_path(window_size, vowel), x=px, y=py)
            else:
                for vowel in VOWEL_TABLE.keys():
                    words = CorpusReader.read_words(vowel, count)
                    px, py = self.create_preprocessor(
                        count, window_size, vowel).make_windows(words)
                    np.savez(self.create_path(window_size, vowel), x=px, y=py)

                file = open(
                    os.path.join(RESOURCE_DIRECTORY, 'prepared',
                                 self.preprocessor,
                                 str(window_size), '_count.txt'), 'w')
                file.write(str(count))
        # else:
        #     parent_path = self.create_parent_path()

        #     sentences = CorpusReader.read_sentences(count)
        #     # TODO rename
        #     characters, tags = process_for_lstm_sequence_tagging_with_accents(
        #         sentences)
        #     path = os.path.join(parent_path, 'prepared')
        #     np.savez(path, x=characters, y=tags)

    def create_preprocessor(self, count, window_size, vowel):
        if self.preprocessor == 'lstm_baseline':
            return LstmBaselinePreprocessor(count, window_size, vowel)
        elif self.preprocessor == 'feedforward':
            return FeedforwardPreprocessor(count, window_size, vowel)

    # TODO
    def create_path(self, *args):
        path = os.path.join(RESOURCE_DIRECTORY, 'prepared', self.preprocessor)
        for arg in args:
            path = os.path.join(path, str(arg))

        parent_path, _ = os.path.split(path)
        os.makedirs(parent_path, exist_ok=True)

        return path

    def create_parent_path(self, *args):
        parent_path = os.path.join(RESOURCE_DIRECTORY, 'prepared',
                                   self.preprocessor)
        for arg in args:
            parent_path = os.path.join(parent_path, str(arg))

        os.makedirs(parent_path, exist_ok=True)
        return parent_path