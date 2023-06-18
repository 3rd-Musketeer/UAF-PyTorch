import sys


class TerminalLogger(object):
    def __init__(self, save_dir='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(save_dir, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def close(self):
        self.log.close()

# sys.stdout = Logger('a.log', sys.stdout)
# sys.stderr = Logger('a.log_file', sys.stderr)
