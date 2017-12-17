'''
Preprocess data for the sequence tagging lstm network
'''

# import time

from src.preprocess.common import normalize_character, deaccentize
from src.preprocess.character_encoder import EnglishEncoder, HungarianEncoder

english_encoder = EnglishEncoder()
hungarian_encoder = HungarianEncoder()


def process(sentences):
    '''Processes the given sentences'''

    characters_by_sentences = []
    tags_by_sentences = []

    # start_time = time.perf_counter()

    for sentence in sentences:

        characters, tags = _process_sentence(sentence)

        characters_by_sentences.append(characters)
        tags_by_sentences.append(tags)

        # if len(characters_by_sentences) % 100 == 0:

        # current_time = time.perf_counter()
        # elapsed_time = current_time - start_time

        # print('processed: ' + str(len(characters_by_sentences)) +
        #       '\telapsed time: ' + str(elapsed_time) + 's')

        # start_time = current_time

    return characters_by_sentences, tags_by_sentences


def _process_sentence(sentence):
    '''Normalizes, deaccents, encodes the sentence and create tags for all characters'''

    characters = []
    tags = []

    for character in sentence:

        normalized = normalize_character(character.lower())
        deaccented = deaccentize(normalized)
        encoded = english_encoder.transform(deaccented)

        tag = hungarian_encoder.transform(normalized)

        characters.append(encoded)
        tags.append(tag)

    return characters, tags
