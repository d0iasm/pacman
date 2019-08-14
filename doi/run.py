import argparse
import collections
import math
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix


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


def result(y, threshold):
    # XSS is 1 and normal is 0.
    print('RESULT: prediction sum(), length of prediction, threshold')
    print(y.sum())
    print(len(y))
    print(threshold)
    print('-----------------------')
    return 1 if y.sum()/len(y) >= threshold else 0


def run():

    normal_data, normal_label = data_loader(NORMAL_TRAIN_FILE, 'normal')
    xss_data, xss_label = data_loader(XSS_TRAIN_FILE, 'xss')
    training_data = [normal_data, xss_data]
    vec = vectorize(training_data)

    threshold = len(xss_data) / (len(xss_data) + len(normal_data))

    X_train = [n for _, n in vec]
    print('-------')
    print("Training:", len(X_train), X_train)
    print("  vector: ", vec)
    print('-------')
    X_train = np.array(X_train).reshape(-1, 1)

    y_train = normal_label + xss_label

    model = GaussianNB()
    model.fit(X_train, y_train)

    # Test with the X_train data.
    print(model.score(X_train, y_train))

    # Test
    xss_test_data, xss_test_label = data_loader(XSS_TEST_FILE, 'xss')
    normal_test_data, normal_test_label = data_loader(NORMAL_TEST_FILE, 'normal')
    y_test = xss_test_label + normal_test_label

    X_test = vectorize(training_data + xss_test_data)
    X_test = np.array([n for _, n in X_test], dtype=object).reshape(-1, 1)
    pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(
        pred, y_test, labels=['xss', 'normal']
    )
    print("acc: \n", acc_score)
    print("confusion matrix: \n", conf_mat)


if __name__ == '__main__':
    run()
