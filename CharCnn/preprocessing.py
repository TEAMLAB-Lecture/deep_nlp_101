import numpy as np
import pandas as pd


def load_data(train_dir, test_dir):
    train_data = pd.read_csv(train_dir)
    test_data = pd.read_csv(test_dir)

    train_x = train_data["data"]
    train_y = train_data["target"]

    test_x = test_data["data"]
    test_y = test_data["target"]

    return train_x, train_y, test_x, test_y


def alphabet_dict():
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    dict = {}
    for i, c in enumerate(alphabet):
        dict[c] = i + 1
    return dict


def strToIndexs(s, length = 300):
    dict = alphabet_dict()
    s = s.lower()
    m = len(s)
    n = min(m, length)
    str2idx = np.zeros(length, dtype='int64')
    for i in range(1, n + 1):
        c = s[-i]
        if c in dict:
            str2idx[i - 1] = dict[c]
    return str2idx


def convert_str2idx(x_data):
    char_idx = []
    for sentences in x_data:
        char_idx.append(strToIndexs(sentences))
    return char_idx


def one_hot(y_data):
    labels = sorted(list(set(y_data.tolist())))
    one_hot = np.zeros((len(labels), len(labels)), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    y_train_raw = y_data.apply(lambda y: label_dict[y]).tolist()
    y_train = np.array(y_train_raw)
    return y_train


