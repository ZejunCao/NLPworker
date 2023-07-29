#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/7/29 11:30
# @File     : decision_tree.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : 决策树实现

# TODO 决策树连续值处理, 其他种类决策树（目前只实现ID3）

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


class TreeNode:
    def __init__(self, feature_index=-1, label=-1):
        self.feature_index = feature_index
        self.label = label
        self.next = {}


class Decision_Tree:
    def __init__(self):
        pass

    def fit(self, input, label):
        input = self.to_numpy(input)
        label = self.to_numpy(label)

        self.n_target = len(set(label))
        self.feature_num = input.shape[-1]
        self.root = self.ID3(input, label)

    # 递归创建树
    def ID3(self, input, label, already_index=[]):
        '''
        D代表整个数据集，特征分别为A_1、A_2、、、A_n，假设第i个特征的所有互异取值为x^i_1、x^i_2、x^i_3、、、x^i_n
        H(D) = -∑(p_i * log2(p_i))   这里的p_i代表第i个标签的统计概率

        g(D, A_i) = H(D) - H(D|A_i) = H(D) - ∑(|x^i_j| / |D| * H(D_j))
        i代表第i个特征，j代表第i个特征的第j种取值，共有n种取值，|x^i_j|代表第i个特征的第j个取值个数，|D|代表数据集D的个数

        H(D_j) = -∑(p_i * log2(p_i))   这里的p_i = x^i_j中标签为k的个数 / x^i_j的个数  k=(0,1,2...)
        :param input: 输入训练数据，不会出现空的情况
        :param label:
        :param already_index: 记录已经进行过决策的特征
        :return: node
        '''
        node = TreeNode()
        cur_n_target = len(set(label))
        # 将当前节点的标签值设为所含样本最多的类别，不管子节点还是父节点，防止出现训练数据之外的特征值
        label_num = [sum(label == i) for i in range(self.n_target)]
        node.label = label_num.index(max(label_num))
        if cur_n_target == 1:
            node.label = label[0]
            return node
        if len(already_index) == self.feature_num:
            return node

        H_D = [sum(label == c) / len(label) for c in set(label)]
        H_D = sum([-p * np.log2(p) for p in H_D])
        g_D_A = []
        for i in range(input.shape[-1]):
            if i in already_index:
                g_D_A.append(0)
                continue
            counter = Counter(input[:, i])
            H_D_Ai = 0
            for k, v in counter.items():
                H_D_j = [sum(label[input[:, i] == k] == j) / v for j in range(cur_n_target)]
                H_D_j = sum([-p * np.log2(p) if p != 0 else 0 for p in H_D_j])
                H_D_Ai += v / len(input) * H_D_j
            g_D_A.append(H_D - H_D_Ai)
        max_gain = g_D_A.index(max(g_D_A))
        node.feature_index = max_gain
        counter = Counter(input[:, max_gain])
        for k in counter.keys():
            node.next[k] = self.ID3(input[input[:, max_gain] == k], label[input[:, max_gain] == k],
                                    already_index + [max_gain])
        return node

    def classify(self, input_test, label_test):
        input_test = self.to_numpy(input_test)
        label_test = self.to_numpy(label_test)
        pred = []
        for input_i in range(len(input_test)):
            node = self.root
            while True:
                if not node.next:
                    pred.append(node.label)
                    break
                # 对于未出现过训练集中的新特征值，直接使用父节点的标签，不再向下递归
                if input_test[input_i][node.feature_index] in node.next:
                    node = node.next[input_test[input_i][node.feature_index]]
                else:
                    pred.append(node.label)
                    break
        accuracy = self.accuracy(pred, label_test)
        return accuracy

    def to_numpy(self, x):
        if isinstance(x, list):
            x = np.array(x)
        return x

    def accuracy(self, pred, label):
        pred = self.to_numpy(pred)
        label = self.to_numpy(label)

        if len(pred.shape) != 1:
            pred = np.argmax(pred, axis=-1)
        return sum(pred == label) / pred.shape[0]


if __name__ == '__main__':
    iris = load_iris()

    X = iris.data
    y = iris.target
    print(X.data.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    # X_train = [['青年', '否', '否', '一般'],
    #            ['青年', '否', '否', '好'],
    #            ['青年', '是', '否', '好'],
    #            ['青年', '是', '是', '一般'],
    #            ['青年', '否', '否', '一般'],
    #            ['中年', '否', '否', '一般'],
    #            ['中年', '否', '否', '好'],
    #            ['中年', '是', '是', '好'],
    #            ['中年', '否', '是', '非常好'],
    #            ['中年', '否', '是', '非常好'],
    #            ['老年', '否', '是', '非常好'],
    #            ['老年', '否', '是', '好'],
    #            ['老年', '是', '否', '好'],
    #            ['老年', '是', '否', '非常好'],
    #            ['老年', '否', '否', '一般'],
    #            ]
    # y_train = [0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
    tree = Decision_Tree()
    tree.fit(X_train, y_train)
    scores = tree.classify(X_test, y_test)
    print(scores)
