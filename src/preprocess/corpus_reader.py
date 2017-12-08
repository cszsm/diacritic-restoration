import os.path

VOWEL_TABLE = {
    'a': ['a', 'á'],
    'e': ['e', 'é'],
    'i': ['i', 'í'],
    'o': ['o', 'ó', 'ö', 'ő'],
    'u': ['u', 'ú', 'ü', 'ű']
}
RESOURCE_DIRECTORY = 'res'

SEQUENCE_LEN = 300


class CorpusReader:
    @staticmethod
    def read_words(vowel, count):
        corpus = open(
            os.path.join(RESOURCE_DIRECTORY, "corpus"), encoding="utf8")
        words = []
        accent_counter = {}

        accents = VOWEL_TABLE[vowel]
        for accent in accents:
            accent_counter[accent] = 0

        for i in range(4):
            next(corpus)

        line_counter = 0
        for line in corpus:
            line_counter += 1
            enough = True

            splits = line.split()
            if splits != []:
                word = splits[0]
                words.append(word)

                for accent in accents:
                    accent_counter[accent] += word.count(accent)
                    if accent_counter[accent] < count:
                        enough = False

                if enough:
                    return words

            if line_counter % 10000 == 0:
                print('line: ' + str(line_counter))
                for accent in accents:
                    print("\t" + accent + ": " + str(accent_counter[accent]))

        print("ERROR: Corpus has run out of words!")
        return words


def read_sentences():
    with open(
            os.path.join('res', 'corpus'), encoding='utf8') as corpus, open(
                os.path.join('res', 'sentences'), encoding='utf8',
                mode='w') as out:

        # sentence_counter = 0
        sentence = ''
        sentences = []
        # sentence_lengths = []

        # true if the current word is in a quote
        quotation_flag = False
        # true if there should be space before the current word
        space_before_flag = False
        # true if there should be space after the current word
        space_after_flag = True

        # i = 0

        for line in corpus:

            if line == '\n':
                # i += 1
                # if i < 14:
                #     continue
                # print(sentence + '\n')

                sentence_length = len(sentence)
                if sentence_length > 0 and sentence_length <= SEQUENCE_LEN:

                    sentences.append(sentence)
                    # sentence_counter += 1
                    out.write(sentence + '\n')

                # sentence_lengths.append(len(sentence))
                sentence = ''

                space_before_flag = False

                prepared_count = len(sentences)
                # if sentence_counter >= count:
                if prepared_count % 100 is 0:
                    print('read: ' + str(prepared_count))

                # if prepared_count >= 20:
                #     return sentences

                continue

            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue

            word, tag = parts

            # deciding where space should be added
            if 'PUNCT' in tag:
                if word is '"':
                    if quotation_flag:
                        quotation_flag = False
                        space_before_flag = False
                    else:
                        quotation_flag = True
                        space_after_flag = False
                elif word in '.,);:!?':
                    space_before_flag = False
                elif word in '(':
                    space_after_flag = False

            # adding space before the current word
            if space_before_flag:
                sentence += ' '
                space_before_flag = space_after_flag
                space_after_flag = True
            else:
                space_before_flag = True

            sentence += word

        # print('ERROR: Not enough sentences in the corpus')
        # return sentences


read_sentences()
