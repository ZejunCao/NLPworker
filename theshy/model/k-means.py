#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/8/16 16:34
# @File     : k-means.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : k均值算法实现

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class KMeans:
    def __init__(self):
        pass

    def fit(self, inputs, n_clusters, init='random'):
        '''

        :param inputs: 输入样本
        :param n_clusters: 要聚类的簇数，即k个聚类中心
        :param init: 初始化k个中心，'random'随机初始化，取前k个样本点作为k个中心
        :return:
        '''
        if init == 'random':
            self.center_clusters = inputs[:n_clusters]
        else:
            raise '请指定一种初始化方法'

        # 存储距离每个聚类中心最近的样本
        self.clusters = [[] for _ in range(n_clusters)]
        # 判断是否继续迭代
        continue_iter = True
        while continue_iter:
            # 存储上一轮记录的聚类样本，当不再改变时，迭代终止
            clusters_pre = self.clusters
            self.clusters = [[] for _ in range(n_clusters)]
            for i in range(len(inputs)):
                min_dis = self.euclidean_distance(inputs[i], self.center_clusters[0])
                label = 0
                for k in range(1, len(self.center_clusters)):
                    dis = self.euclidean_distance(inputs[i], self.center_clusters[k])
                    if dis < min_dis:
                        min_dis = dis
                        label = k
                self.clusters[label].append(inputs[i].tolist())

            # 更新聚类中心
            self.center_clusters = [np.array(c).mean(axis=0) for c in self.clusters]

            for i in range(len(self.clusters)):
                if len(self.clusters[i]) == len(clusters_pre[i]):
                    a = np.array(self.clusters[i]) == np.array(clusters_pre[i])
                    b = a.all()
                if len(self.clusters[i]) != len(clusters_pre[i]) or not (np.array(self.clusters[i]) == np.array(clusters_pre[i])).all():
                    break
            else:
                continue_iter = False
        for i in range(len(self.clusters)):
            self.clusters[i] = np.array(self.clusters[i])
        self.center_clusters = np.array(self.center_clusters)
        return

    def euclidean_distance(self, x1, x2):
        assert len(x1) == len(x2), f'请输入相同维度的两个点，x1_len: {len(x1)}, x2_len: {len(x2)}'
        distance = np.sqrt(sum([(x1[i] - x2[i]) ** 2 for i in range(len(x1))]))
        return distance


if __name__ == '__main__':

    # make_blobs：生成聚类的数据集
    # n_samples：生成的样本点个数，n_features：样本特征数，centers：样本中心数
    # cluster_std：聚类标准差，shuffle：是否打乱数据，random_state：随机种子
    X, _ = make_blobs(n_samples=150, n_features=2, centers=3,cluster_std=0.5,shuffle=True, random_state=0)
    kmeans = KMeans()
    kmeans.fit(X, n_clusters=3)

    #散点图
    #c:点得颜色,maker：点的形状,edgecolor:点边缘的形状，s:点的大小
    # plt.scatter(X[:,0], X[:,1], c='white', marker = 'o', edgecolors='black', s=50)
    # plt.show()

    # 画出预测的三个簇类
    plt.scatter(
        kmeans.clusters[0][:, 0], kmeans.clusters[0][:, 1],
        s = 50, c = 'orange',
        marker= 'o', edgecolors='black',
        label = 'cluster 1'
    )

    plt.scatter(
        kmeans.clusters[1][:, 0], kmeans.clusters[1][:, 1],
        s = 50, c = 'orange',
        marker= 'o', edgecolors='black',
        label = 'cluster 1'
    )

    plt.scatter(
        kmeans.clusters[2][:, 0], kmeans.clusters[2][:, 1],
        s=50, c='lightblue',
        marker='v', edgecolor='black',
        label='cluster 3'
    )

    #画出聚类中心
    plt.scatter(
        kmeans.center_clusters[:, 0], kmeans.center_clusters[:, 1],
        s = 250, marker= '*',
        c = 'red', edgecolors='black',
        label = 'centroids'
    )
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()
    print()
