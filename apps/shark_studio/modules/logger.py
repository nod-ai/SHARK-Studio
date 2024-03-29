import sys


class Logger:
    def __init__(self, filename, filter=None):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.filter = filter

    def write(self, message):
        for x in message.split("\n"):
            if self.filter in x:
                self.log.write(message)
            else:
                self.terminal.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def logger_test(x):
    print("[LOG] This is a test")
    print(f"This is another test, without the filter")
    return x


def read_sd_logs():
    sys.stdout.flush()
    with open("shark_tmp/sd.log", "r") as f:
        return f.read()


sys.stdout = Logger("shark_tmp/sd.log", filter="[LOG]")
