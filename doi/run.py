import argparse
import collections
import math
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


# Level 1
XSS_TRAIN_FILE = 'dataset/train_level_1.csv'
XSS_TEST_FILE = 'dataset/test_level_1.csv'
NORMAL_TRAIN_FILE = 'dataset/normal.csv'
NORMAL_TEST_FILE = 'dataset/normal.csv'

# Level 2
XSS_TRAIN_FILE_2 = 'dataset/train_level_2.csv'
XSS_TEST_FILE_2 = 'dataset/test_level_2.csv'

STOP_WORDS = [';']
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
    t = t.replace("\'", "")
    t = t.replace("\"", "")
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

    """
    データ前処理・学習機作成
    """
    X_train = xss_train_data + normal_train_data + xss_train_data_2
    y_train = xss_train_label + normal_train_label + xss_train_label_2
    X_test = xss_test_data_2 + normal_test_data
    y_test = xss_test_label_2 + normal_test_label

    vec = TfidfVectorizer()
    X_train = vec.fit_transform(X_train)
    X_train = X_train.todense()

    #model = BernoulliNB()
    model = GaussianNB()
    #model = MultinomialNB()
    #model = ComplementNB()
    model.fit(X_train, y_train)

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
    print("acc: \n", acc_score)
    print("confusion matrix: \n", conf_mat)


if __name__ == '__main__':
    run()
