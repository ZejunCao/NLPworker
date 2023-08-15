#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/8/14 21:05
# @File     : knn.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : KNN算法实现，实现KD树


import numpy as np

class TreeNode:
    def __init__(self, sample=0, left=None, right=None, split_index=0):
        self.sample = sample
        self.left = left
        self.right = right
        # 按第几个维度划分的
        self.split_index = split_index

class KNN:
    def __init__(self):
        self.kdtree = None
        self.min_distance = -1
        self.nn_sample = None
        # 寻找最近邻点时分为两个阶段，前向找到最近的叶子节点，然后回溯寻找更近的节点，self.backtrack记录前向和回溯的状态
        self.backtrack = False

    def fit(self, inputs):
        inputs = self.to_numpy(inputs)
        self.kdtree = self.KDTree(inputs)
        print()

    def KDTree(self, inputs, dim=0):
        dim = dim % inputs.shape[1]
        node = TreeNode(split_index=dim)
        if len(inputs) == 1:
            node.sample = inputs[0]
            return node

        inputs = inputs[np.argsort(inputs[:, dim])]
        middle = len(inputs) // 2
        node.sample = inputs[middle]
        if len(inputs[:middle]) != 0:
            node.left = self.KDTree(inputs[:middle], dim=dim+1)
        if len(inputs[middle + 1:]) != 0:
            node.right = self.KDTree(inputs[middle + 1:], dim=dim+1)
        return node

    def find_nn(self, x, node=None):
        '''
        查找最近邻
        :param x:
        :return:
        '''
        if not node:
            return

        dim = node.split_index
        if x[dim] < node.sample[dim]:
            self.find_nn(x, node.left)
        else:
            self.find_nn(x, node.right)

        if self.min_distance == -1:
            self.nn_sample = node.sample
            self.min_distance = self.euclidean_distance(x, node.sample)
            return
        if abs(x[dim] - node.sample[dim]) <= self.min_distance:
            dis = self.euclidean_distance(x, node.sample)
            if dis < self.min_distance:
                self.min_distance = dis
                self.nn_sample = node.sample

            if x[dim] < node.sample[dim]:
                self.find_nn(x, node.right)
            else:
                self.find_nn(x, node.left)


    def euclidean_distance(self, x1, x2):
        assert len(x1) == len(x2), f'请输入相同维度的两个点，x1_len: {len(x1)}, x2_len: {len(x2)}'
        distance = np.sqrt(sum([(x1[i] - x2[i]) ** 2 for i in range(len(x1))]))
        return distance


    def to_numpy(self, x):
        if isinstance(x, list):
            x = np.array(x)
        return x


if __name__ == '__main__':
    dataSet = [[2, 3],
               [5, 4],
               [9, 6],
               [4, 7],
               [8, 1],
               [7, 2]]
    x = [3, 4.5]
    knn = KNN()
    knn.fit(dataSet)
    knn.find_nn(x, knn.kdtree)
    print('x最近邻为：', knn.nn_sample)
    print('距离为：', knn.min_distance)
    print()
