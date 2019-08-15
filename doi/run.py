import argparse
import collections
import math
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Level 1
XSS_TRAIN_FILE = 'dataset/train_level_1.csv'
XSS_TEST_FILE = 'dataset/test_level_1.csv'
NORMAL_TRAIN_FILE = 'dataset/normal.csv'
NORMAL_TEST_FILE = 'dataset/normal.csv'

# Level 2
XSS_TRAIN_FILE_2 = 'dataset/train_level_2.csv'
XSS_TEST_FILE_2 = 'dataset/test_level_2.csv'

# Level 4
NORMAL_TRAIN_FILE_4 = 'dataset/train_level_4.csv'
NORMAL_TEST_FILE_4 = 'dataset/test_level_4.csv'

STOP_WORDS = [';', '\"', '\'']
BLACK_LISTS = ["alert"]

keys = []
test_src = ""
test_result = ""


def data_loader(f_name, l_name):
    with open(f_name, mode='r', encoding='utf-8') as f:
        data = list(set(f.readlines()))
        label = [l_name for i in range(len(data))]
        return data, label


def clean(t):
    target = ["</", "/>", "<", ">", "=", ":", "/", "(", ")", "[", "]", "{", "}", "＜", "＞"]
    for ch in target:
        t = t.replace(ch, " ")
    return t


def run():
    """
    データ作成
    """
    # Level 1
    xss_train_data, xss_train_label = data_loader(XSS_TRAIN_FILE, 'xss')
    xss_test_data, xss_test_label = data_loader(XSS_TEST_FILE, 'xss')
    normal_train_data, normal_train_label = data_loader(NORMAL_TRAIN_FILE, 'normal')
    normal_test_data, normal_test_label = data_loader(NORMAL_TEST_FILE, 'normal')

    # Level 2
    xss_train_data_2, xss_train_label_2 = data_loader(XSS_TRAIN_FILE_2, 'xss')
    xss_test_data_2, xss_test_label_2 = data_loader(XSS_TEST_FILE_2, 'xss')

    # Level 4
    normal_train_data_4, normal_train_label_4 = data_loader(NORMAL_TRAIN_FILE_4, 'normal')
    normal_test_data_4, normal_test_label_4 = data_loader(NORMAL_TEST_FILE_4, 'normal')

    """
    データ前処理・学習機作成
    """
    X_train = xss_train_data + normal_train_data + xss_train_data_2 + normal_train_data_4
    y_train = xss_train_label + normal_train_label + xss_train_label_2 + normal_train_label_4
    X_test = normal_test_data_4 + xss_test_data
    y_test = normal_test_label_4 + xss_test_label

    vec = TfidfVectorizer(preprocessor=clean, stop_words=STOP_WORDS)
    X_train = vec.fit_transform(X_train)
    X_train = X_train.todense()

    param_grid = [
        {
            "kernel": ["linear"],
            "gamma": [x/10.0 for x in range(10)],
            "degree": [x for x in range(10)],
            "coef0": [x/10.0 - 5.0 for x in range(10)],
        },
        {
            "kernel": ["poly"],
            "gamma": [x/10.0 for x in range(10)],
            "degree": [x for x in range(10)],
            "coef0": [x/10.0 - 5.0 for x in range(10)],
        },
        {
            "kernel": ["rbf"],
            "gamma": [x/10.0 for x in range(10)],
            "degree": [x for x in range(10)],
            "coef0": [x/10.0 - 5.0 for x in range(10)],
        },
        {
            "kernel": ["sigmoid"],
            "gamma": [x/10.0 for x in range(10)],
            "degree": [x for x in range(10)],
            "coef0": [x/10.0 - 5.0 for x in range(10)],
        },
    ]

    model = GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring='accuracy', cv=5)
    model.fit(X_train, y_train)
    print('=============')
    print(model.best_params_)
    print('=============')

    """
    テスト
    """
    X_test = vec.transform(X_test)
    X_test = X_test.todense()
    pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, pred)
    conf_mat = confusion_matrix(
        pred, y_test, labels=['xss', 'normal']
    )

    idx = 0
    for p, a in zip(pred, y_test):
        if p != a:
          print("-----------")
          print("ans: ", a, " pred:", p)
          print((normal_test_data_4 + xss_test_data)[idx])
        idx += 1

    print("acc: \n", acc_score)
    print("confusion matrix: \n", conf_mat)


if __name__ == '__main__':
    run()
