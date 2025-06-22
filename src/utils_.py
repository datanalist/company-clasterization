import numpy as np


def save_np_tesors(np_tesors, path):
    np.save(path, np_tesors)


def load_np_tesors(path):
    return np.load(path)
