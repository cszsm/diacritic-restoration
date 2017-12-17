import os.path
from collections import Counter

def get_sentence_lengths():
    with open(os.path.join('res', 'testcorpus'), encoding="utf8") as corpus:

        # sentence_counter = 0
        sentence = ''
        sentence_lengths = []

        # true if the current word is in a quote
        quotation_flag = False
        # true if there should be space before the current word
        space_before_flag = False
        # true if there should be space after the current word
        space_after_flag = True

        for line in corpus:

            if line == '\n':
                # print(sentence + '\n')

                sentence_lengths.append(len(sentence))
                sentence = ''
                
                space_before_flag = False

                # sentence_counter += 1
                # if sentence_counter >= 5:
                #     break

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
                elif word in '.,);:':
                    space_before_flag = False

            # adding space before the current word
            if space_before_flag:
                sentence += ' '
                space_before_flag = space_after_flag
                space_after_flag = True
            else:
                space_before_flag = True

            sentence += word

        return sentence_lengths

def get_punctuations():
    with open(os.path.join('res', 'corpus'), encoding="utf8") as corpus:

        punctuations = set()

        for line in corpus:

            if line == '\n':
                continue
            
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            
            word, tag = parts
            
            if 'PUNCT' in tag:
                punctuations.add(word)

        return punctuations
