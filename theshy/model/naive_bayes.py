#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/7/28 14:42
# @File     : naive_bayes.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : 朴素贝叶斯实现

# TODO 贝叶斯连续值处理

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


class Naive_Bayes:
    def __init__(self):
        '''
        self.prior_prob存储所有先验概率，形式为
        {
            0:{
                [P(Y=0),
                {P(X^1=X_1|Y=0):value, P(X^1=X_2|Y=0):value, P(X^1=X_3|Y=0):value},
                {P(X^2=X_1|Y=0):value, P(X^2=X_2|Y=0):value, P(X^2=X_3|Y=0):value},
                {P(X^3=X_1|Y=0):value, P(X^3=X_2|Y=0):value, P(X^3=X_3|Y=0):value},
                {P(X^4=X_1|Y=0):value, P(X^4=X_2|Y=0):value, P(X^4=X_3|Y=0):value},
                ]
            }
            1:{
                [P(Y=0),
                {P(X^1=X_1|Y=1):value, P(X^1=X_2|Y=1):value, P(X^1=X_3|Y=1):value},
                {P(X^2=X_1|Y=1):value, P(X^2=X_2|Y=1):value, P(X^2=X_3|Y=1):value},
                {P(X^3=X_1|Y=1):value, P(X^3=X_2|Y=1):value, P(X^3=X_3|Y=1):value},
                {P(X^4=X_1|Y=1):value, P(X^4=X_2|Y=1):value, P(X^4=X_3|Y=1):value},
                ]
            }
        }
        '''
        self.prior_prob = []
        self.lambda_ = 1  # 平滑项，取1为拉普拉斯平滑
        self.oov_prob = []  # 测试集中若出现训练集中未出现过的特征，直接使用平滑最小值

    def fit(self, input, label):
        '''
        统计方法，通过训练数据直接得到先验概率
        使用拉普拉斯平滑，P(X_1=1.5|Y=0)=(∑(X_1=1.5,Y=0) + lambda_) / (∑(Y=0) + 所有训练集中X种类数 * lambda_)
        '''
        self.n_target = len(set(label))
        n_label = Counter(label)
        # 得到P(Y=0)、P(Y=1)
        self.prior_prob = {k: [(v + self.lambda_) / (len(label) + self.n_target * self.lambda_)] for k, v in n_label.items()}
        # self.
        for i in range(self.n_target):
            data_i = input[label == i]
            for j in range(data_i.shape[-1]):
                counter = Counter(data_i[:, j])
                p_x = {k: (counter[k] + self.lambda_) / (len(data_i) + len(set(input[:, j])) * self.lambda_) for k in set(input[:, j])}
                # a = sum(p_x.values())
                self.prior_prob[i].append(p_x)
                if i == 0:
                    self.oov_prob.append(self.lambda_ / (len(data_i) + len(set(input[:, j])) * self.lambda_))

    def classify(self, input_test, label_test):
        pred_prob = []
        # 遍历每个样本
        for input_i in range(len(input_test)):
            # 计算每个标签的概率
            y_prob = []  # 存储该样本每个标签的概率
            for label_i in range(self.n_target):
                prob = 1
                # 取出每个特征的概率
                for feature_i in range(input_test.shape[-1]):
                    x = input_test[input_i][feature_i]
                    prob = prob * self.prior_prob[label_i][feature_i+1].get(x, self.oov_prob[feature_i])
                y_prob.append(prob)
            pred_prob.append(y_prob)
        pred = np.argmax(np.array(pred_prob), axis=-1)
        accuracy = self.accuracy(pred, label_test)
        return accuracy

    def accuracy(self, pred, label):
        if isinstance(pred, list):
            pred = np.array(pred)
        if isinstance(label, list):
            label = np.array(label)

        if len(pred.shape) != 1:
            pred = np.argmax(pred, axis=-1)
        return sum(pred == label) / pred.shape[0]


if __name__ == '__main__':
    iris = load_iris()

    X = iris.data
    y = iris.target
    print(X.data.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    Bayes = Naive_Bayes()
    Bayes.fit(X_train, y_train)
    scores = Bayes.classify(X_test, y_test)
    print(scores)