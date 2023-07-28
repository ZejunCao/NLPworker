#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/7/25 22:27
# @File     : logistic_regression.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : 逻辑回归实现


from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Logistic_Regression:
    def __init__(self, optimizer='GD', lr=0.001, max_iterations=1000):
        self.optimizer = optimizer
        self.lr = lr
        self.max_iterations = max_iterations

    def fit(self, input, label, input_test, label_test, n_target=2):
        self.n_target = n_target
        # 多分类,使用softmax
        if self.n_target > 2:
            self.weights = np.random.normal(0, 0.1, (input.shape[1], self.n_target))
            self.bias = np.zeros(self.n_target)

            # 梯度下降法求解
            if self.optimizer == 'GD':
                for iteration in range(self.max_iterations):
                    pred = np.dot(input, self.weights) + self.bias
                    pred = self.softmax(pred)
                    accuracy = self.accuracy(pred, label)
                    loss = self.cross_entropy_multi(pred, label)
                    print(f'{iteration}, accuracy: {accuracy}, loss:{loss}')

                    label_expand = np.array([[0] * l + [1] + [0] * (self.n_target - 1 - l) for l in label])
                    softmax_grad = []
                    for sample in range(label_expand.shape[0]):
                        softmax_grad.append([[-label_expand[sample][i]*(1-pred[sample][j]) if i == j else label_expand[sample][i]*pred[sample][j] for j in range(self.n_target)] for i in range(self.n_target)])
                    softmax_grad = np.array(softmax_grad)
                    input_repeat = np.expand_dims(input, axis=-1).repeat(3, axis=-1)

                    w_grad = np.matmul(input_repeat, softmax_grad).mean(axis=0)
                    bias_grad = (softmax_grad.sum(axis=0)).mean(axis=0)

                    self.weights -= self.lr * w_grad
                    self.bias -= self.lr * bias_grad

                    if (iteration + 1) % 500 == 0:
                        self.test(input_test, label_test)
                        print(f'{iteration + 1}, accuracy: {accuracy}')
        # 二分类,使用sigmoid
        else:
            self.weights = np.random.normal(0, 0.1, (input.shape[1]))
            self.bias = 0

            # 梯度下降法求解
            if self.optimizer == 'GD':
                for iteration in range(self.max_iterations):
                    pred = np.dot(input, self.weights) + self.bias
                    pred = self.sigmoid(pred)  # pred预测的值代表标签为1的概率
                    pred_class = (pred > 0.5) + 0
                    accuracy = self.accuracy(pred_class, label)
                    loss = self.cross_entropy_binary(pred, label)
                    print(f'{iteration}, accuracy: {accuracy}, loss:{loss}')

                    w_grad = (1 / input.shape[0]) * np.matmul(input.T, pred - label)
                    bias_grad = (pred - label).mean()

                    self.weights -= self.lr * w_grad
                    self.bias -= self.lr * bias_grad

                    if (iteration + 1) % 10 == 0:
                        self.test(input_test, label_test)
                        print(f'{iteration + 1}, accuracy: {accuracy}')
        return

    def test(self, input_test, label_test):
        pred = np.dot(input_test, self.weights) + self.bias
        if self.n_target > 2:
            pred = self.softmax(pred)
        else:
            pred = self.sigmoid(pred)  # pred预测的值代表标签为1的概率
            pred = (pred > 0.5) + 0
        accuracy = self.accuracy(pred, label_test)
        return accuracy

    def softmax(self, x):
        return np.exp(x) / np.expand_dims(np.exp(x).sum(axis=1), axis=-1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def cross_entropy_multi(self, pred, label):
        loss = pred[range(pred.shape[0]), label] * np.log(pred[range(pred.shape[0]), label])
        return -loss.mean()

    def cross_entropy_binary(self, pred, label):
        loss = label * np.log(pred) + (1 - label) * np.log(1 - pred)
        return -loss.mean()

    def accuracy(self, pred, label):
        if len(pred.shape) != 1:
            pred = np.argmax(pred, axis=-1)
        return sum(pred == label) / pred.shape[0]


if __name__ == '__main__':
    iris = load_iris()

    X = iris.data
    y = iris.target
    print(X.data.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=420)
    # 一共150个样本，分别是50个类别1、50个类别2、50个类别3，若想测试二分类可以取前100个样本
    # X_train, X_test, y_train, y_test = train_test_split(X[:100], y[:100], test_size=0.15, random_state=420)
    LR = Logistic_Regression(optimizer='GD', lr=0.5, max_iterations=5000)
    LR.fit(X_train, y_train, X_test, y_test, n_target=3)

    # lrl1 = LR(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
    # # lrl2 = LR(penalty="l2", solver="liblinear", C=0.5, max_iter=1000)
    # lrl1 = lrl1.fit(X, y)
    # print(lrl1.coef_)  # coef_查看每个特征所对应的参数
    # print((lrl1.coef_ != 0).sum(axis=1))  # array([10]),30个特征中有10个特征的系数不为0;由此可见l1正则化会让参数的系数为0
    # # lrl2 = lrl2.fit(X, y)
    # # print(lrl2.coef_)  # 没有一个参数的系数为0,由此可见l2会尽量让每一个参数都能有贡献
    # l1 = []
    # # l2 = []
    # l1test = []
    # # l2test = []
    # for i in np.linspace(0.05, 1.5, 19):  # 取了19个数
    #     lrl1 = LR(penalty="l1", solver="liblinear", C=i, max_iter=1000)
    #     lrl2 = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)
    #     lrl1 = lrl1.fit(X_train, y_train)  # 对模型训练
    #     l1.append(accuracy_score(lrl1.predict(X_train), y_train))  # 训练的结果
    #     l1test.append(accuracy_score(lrl1.predict(X_test), y_test))  # 测试的结果
    # print(l1)
    # print(l1test)
    # graph = [l1, l2, l1test, l2test]
    # color = ["green", "black", "lightgreen", "gray"]
    # label = ["L1", "L2", "L1test", "L2test"]
    # plt.figure(figsize=(6, 6))
    # for i in range(len(graph)):
    #     plt.plot(np.linspace(0.05, 1.5, 19), graph[i], color[i], label=label[i])#折线图
    # plt.legend(loc=4)  # 图例的位置在哪里?4表示，右下角
    # plt.show()
