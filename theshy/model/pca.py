#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/8/4 16:43
# @File     : pca.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : 主成分分析,无监督


import numpy as np

class PCA:
    def __init__(self):
        pass

    def fit_(self, x, dim=-1, contribution=-1):
        '''
        使用样本相关矩阵的特征值分解算法实现
        :param x: 输入数据
        :param dim: 转化之后的维度
        :param contribution: 取贡献率大于指定值的主成分，贡献率和维度设置一个即可，先判断维度是否设置
        :return:
        '''
        # 样本数，特征维度
        samples_num, feature_dim = x.shape
        assert dim <= feature_dim, 'pca之后的维度需小于等于输入维度'
        assert contribution == -1 or (contribution <= 1 and contribution > 0), 'contribution需设置(0, 1]之间'
        # 样本均值
        self.mean_value = x.mean(axis=0)
        # 中心化/零均值化
        x -= self.mean_value
        # 样本协方差矩阵
        S = np.array([[sum([(x[k][i]) * (x[k][j]) for k in range(samples_num)]) / (samples_num - 1)
                       for j in range(feature_dim)] for i in range(feature_dim)])
        # 样本相关矩阵
        R = np.array([[S[i][j] / np.sqrt(S[i][i] * S[j][j]) if np.sqrt(S[i][i] * S[j][j]) != 0 else 0 for j in range(feature_dim)] for i in range(feature_dim)])

        # 特征值和特征向量,特征向量是按列看的，要转置
        eigenvalue, featurevector = np.linalg.eig(R)

        sort_index = np.argsort(eigenvalue)[::-1]
        eigenvalue = eigenvalue[sort_index]
        featurevector = featurevector.T[sort_index]

        if dim != -1:
            self.P = featurevector[:dim]
        elif contribution != -1:
            contri_sum = 0
            eigenvalue_sum = sum(eigenvalue)
            for i in range(feature_dim):
                contri_sum += eigenvalue[i]
                if contri_sum / eigenvalue_sum >= contribution:
                    dim = i
                    break
            self.P = featurevector[:dim]
        Y = np.matmul(self.P, x.T)
        return Y

    def fit(self, x, dim=-1, contribution=-1):
        '''
        使用奇异值分解算法实现
        :param x: 输入数据
        :param dim: 转化之后的维度
        :param contribution: 取贡献率大于指定值的主成分，贡献率和维度设置一个即可，先判断维度是否设置
        :return:
        '''
        # 样本数，特征维度
        samples_num, feature_dim = x.shape
        assert dim <= feature_dim, 'pca之后的维度需小于等于输入维度'
        assert contribution == -1 or (contribution <= 1 and contribution > 0), 'contribution需设置(0, 1]之间'
        # 样本均值
        self.mean_value = x.mean(axis=0)
        # 将x类型转换成与self.mean_value相同的类型，防止不能减法运算
        x = x.astype(self.mean_value.dtype)
        # 中心化/零均值化
        x -= self.mean_value

        # 奇异值分解
        U, S, V_T = self.svd(x)

        if dim != -1:
            self.P = V_T[:dim]
        elif contribution != -1:
            contri_sum = 0
            eigenvalue_sum = sum(S)
            for i in range(feature_dim):
                contri_sum += S[i]
                if contri_sum / eigenvalue_sum >= contribution:
                    dim = i
                    break
            self.P = V_T[:dim]
        Y = np.matmul(self.P, x.T)

        return Y

    def svd(self, x):
        '''
        奇异值分解，X.T*X的特征值作为S，特征向量作为V，X*X.T的特征向量作为U
        :param x:
        :return:
        '''
        eigenvalue, featurevector = np.linalg.eig(np.matmul(x.T, x))
        # 这里的特征值是无序的，需要按照从大到小顺序重排
        sort_index = np.argsort(eigenvalue)[::-1]
        S = np.sqrt(eigenvalue[sort_index])
        V_T = featurevector.T[sort_index]

        eigenvalue, featurevector = np.linalg.eig(np.matmul(x, x.T))
        sort_index = np.argsort(eigenvalue)[::-1]
        U = featurevector.T[sort_index].T

        return U, S, V_T


if __name__ == '__main__':
    mat = np.array([[85, 140, 80, 100],
                    [130, 75, 110, 50],
                    [140, 145, 135, 105],
                    [75, 30, 45, 35]])
    # mat = np.array([[10, 140, 80, 100],
    #                 [10, 75, 110, 50],
    #                 [10, 145, 135, 105]])
    # mat = np.array([[ 140, 80, 100],
    #                 [ 75, 110, 50],
    #                 [ 145, 135, 105],
    #                 [ 30, 45, 35]])
    pca = PCA()
    pca_mat = pca.fit(mat, dim=2).T

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    newX = pca.fit_transform(mat)  # 等价于pca.fit(X) pca.transform(X)
    invX = pca.inverse_transform(newX)  # 将降维后的数据转换成原始数据


    print()

