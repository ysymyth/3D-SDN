import os
import time


def new_working_dir(working_dir_root, name=None):
    working_dir = os.path.join(working_dir_root, time.strftime('%Y-%m-%d-%H%M%S'))
    if name is not None:
        working_dir += '-' + name

    os.makedirs(working_dir)
    return working_dir


class Message(object):
    num_level = 0
    delim = '  '

    def __init__(self, message=None):
        self.message = message or ''

    def __enter__(self):
        self.start = time.time()
        print('{:s}{:s} --->'.format(Message.delim * Message.num_level, self.message))

        Message.num_level += 1
        return self

    def __exit__(self, *args):
        Message.num_level -= 1

        print('{:s}---> {:s} [{:.4f} s]'.format(Message.delim * Message.num_level, self.message, time.time() - self.start))
