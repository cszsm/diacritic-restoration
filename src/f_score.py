'''Module for calculating F-score'''


class FScore:
    '''Class for calculating F-score'''

    def __init__(self, true_positives=0, false_positives=0, false_negatives=0):
        self.tp = true_positives
        self.fp = false_positives
        self.fn = false_negatives

    def __iadd__(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

        return self

    def __repr__(self):
        return self._string()

    def __str__(self):
        return self._string()

    def _precision(self):
        if self.tp + self.fp == 0:
            return -1

        return self.tp / (self.tp + self.fp)

    def _recall(self):
        if self.tp + self.fn == 0:
            return -1

        return self.tp / (self.tp + self.fn)

    def calculate_fscore(self):
        '''Calculates F-score'''

        precision = self._precision()
        recall = self._recall()

        if precision == -1 or recall == -1 or precision + recall == 0:
            return -1

        return 2 * ((precision * recall) / (precision + recall))

    def _string(self):
        true_positives = 'true positives: ' + str(self.tp) + '\n'
        false_positives = 'false positives: ' + str(self.fp) + '\n'
        false_negatives = 'false negatives: ' + str(self.fn) + '\n'

        precision = 'precision: ' + str(self._precision()) + '\n'
        recall = 'recall: ' + str(self._recall()) + '\n'

        f_score = 'f-score: ' + str(self.calculate_fscore())

        string = true_positives + false_positives + false_negatives + '\n'
        string += precision + recall + '\n'
        string += f_score

        return string
