'''Preprocess data for the sequence tagging lstm network'''
from src.preprocess.common import tag_character, normalize_character, deaccentize
from src.preprocess.common import fit_encoders, transform

def process_for_train(sentences):
    '''Processes the given sentences'''

    fit_encoders()

    characters_by_sentences = []
    tags_by_sentences = []

    for sentence in sentences:
        characters, tags = process_sentence(sentence)

        characters_by_sentences.append(characters)
        tags_by_sentences.append(tags)

    return characters_by_sentences, tags_by_sentences

def process_sentence(sentence):
    '''Encodes the characters of the sentence and create tags for all characters'''

    characters = []
    tags = []

    for character in sentence:

        normalized = normalize_character(character.lower())
        deaccented = deaccentize(normalized)
        encoded = transform(deaccented)

        tag = tag_character(normalized)

        characters.append(encoded)
        tags.append(tag)

    return characters, tags
