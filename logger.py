class Logger:

    def __init__(self, filename):
        self.file = open(filename, 'w')

    def log(self, text):
        self.file.write(text)