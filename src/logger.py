import os.path

class Logger:

    def __init__(self, filename):
        self.file = open(os.path.join('../logs', filename + '.txt'), 'w')

    def log(self, text):
        print(text)
        self.file.write(text)