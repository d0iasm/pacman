import argparse
import collections
import math
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix

# Level 1
XSS_TRAIN_FILE = 'dataset/train_level_1.csv'
XSS_TEST_FILE = 'dataset/test_level_1.csv'
NORMAL_TRAIN_FILE = 'dataset/normal.csv'
NORMAL_TEST_FILE = 'dataset/normal.csv'

STOP_WORDS = ['']

keys = []
test_src = ""
test_result = ""


def data_loader(src, label):
    data = []
    with open(src) as f:
        for line in f:
            data += clean(line).split()
    return data, [label for _ in range(len(data))]


def clean(t):
    t = t.replace("</", " ")
    t = t.replace("<", " ")
    t = t.replace(">", " ")
    t = t.replace("=", " ")
    t = t.replace("\'", "")
    t = t.replace("\"", "")
    return t


def vectorize(tokens):
    total = sum([len(t) for t in tokens])
    vec = []

    # Count term frequency.
    dicts = []
    for t in tokens:
        d = collections.defaultdict(int)
        for k in t:
            d[k] += 1
        dicts.append(d)

    # Calculate TF-IDF.
    for i, t in enumerate(tokens):
        for term in t:
            # TF(ti, dj) = (The count ti in dj) / (The total count ti in all documents)
            tf = dicts[i][term] / sum([d[term] for d in dicts])

            # IDF(ti) = log(The number of documents / The number of documents that has ti)
            idf = math.log10(total / sum([1 if d[term] > 0 else 0 for d in dicts]) + 1)
            vec.append([term, tf * idf])

    return vec


def run():
    """
    データ作成
    """
    # Level 1
    xss_train_data, xss_train_label = data_loader(XSS_TRAIN_FILE, 'xss')
    xss_test_data, xss_test_label = data_loader(XSS_TEST_FILE, 'xss')
    normal_train_data, normal_train_label = data_loader(NORMAL_TRAIN_FILE, 'normal')
    normal_test_data, normal_test_label = data_loader(NORMAL_TEST_FILE, 'normal')


    """
    データ前処理・学習機作成
    """
    # NOTE: The argument of vectorize() must be a 2d array.
    vec = vectorize([xss_train_data, normal_train_data])

    X_train = [n for _, n in vec]
    X_train = np.array(X_train).reshape(-1, 1)

    y_train = xss_train_label + normal_train_label
    X_test = xss_test_data + normal_test_data
    y_test = xss_test_label + normal_test_label

    model = GaussianNB()
    model.fit(X_train, y_train)

    """
    テスト
    """
    # NOTE: The argument of vectorize() must be a 2d array.
    X_test = vectorize([xss_test_data, normal_test_data])
    X_test = np.array([n for _, n in X_test], dtype=object)
    X_test = X_test.reshape(-1, 1)
    pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(
        pred, y_test, labels=['xss', 'normal']
    )
    print("acc: \n", acc_score)
    print("confusion matrix: \n", conf_mat)


if __name__ == '__main__':
    run()
