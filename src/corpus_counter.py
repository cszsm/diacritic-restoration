import os.path
from collections import Counter

def get_sentence_lengths():
    with open(os.path.join('res', 'corpus'), encoding="utf8") as corpus:

        for i in range(5):
            next(corpus)

        line_counter = 0
        sentence_counter = 0
        sentence = ''
        sentence_lengths = []

        # state flags
        quotation_flag = False
        space_before_flag = True
        space_after_flag = True

        for line in corpus:

            line_counter += 1

            if line == '\n':
                # print(sentence + '\n')

                sentence_lengths.append(len(sentence))

                sentence = ''
                sentence_counter += 1

                # if sentence_counter >= 5:
                #     break

                continue
            
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            
            word, tag = parts

            if sentence is '':
                sentence = word
                continue

            if 'PUNCT' in tag:
                # punctuations found in the corpus: .,()-";:
                if word is '"':
                    if quotation_flag:
                        quotation_flag = False
                        space_after_flag = False
                    else:
                        quotation_flag = True
                        space_before_flag = False
                elif word in '.,);:':
                    space_after_flag = False
            if space_after_flag:
                sentence += ' '
                space_after_flag = space_before_flag
                space_before_flag = True
            else:
                space_after_flag = True

            sentence += word

        return sentence_lengths

get_sentence_lengths()