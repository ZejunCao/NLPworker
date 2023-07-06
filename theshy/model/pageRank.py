#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 16:55
# @File    : crf.py
# @Software: PyCharm
# @System  : Windows
# @desc    : null

import networkx as nx
import numpy as np


# 每个元素非负，每列元素之和为1
# 迭代法
def pagerank(M, d=0.85, max_iter=100, tol=1e-6):
    assert isinstance(M, np.ndarray), '请将M矩阵转为numpy array格式'
    assert len(M.shape) == 2, '确保M矩阵为二维数组'
    assert (M >= 0).all(), '确保M矩阵每个元素都非负'

    n = M.shape[1]
    for i in range(n):
        sum_ = sum(M[:, i])
        assert sum_ != 0, '每列之和不可为0'
        M[:, i] /= sum_

    R_t1 = np.full((n, 1), 1 / n)
    for i in range(max_iter):
        R_t0 = R_t1
        R_t1 = d * np.matmul(M, R_t0) + (1 - d) / n
        if sum(abs(R_t1 - R_t0)) < n * tol:
            break
    else:
        print(f'全部迭代后容差仍未为{sum(abs(R_t1 - R_t0))}')
    return R_t1


# 幂法
def pagerank_power(M, d=0.85, max_iter=100, tol=1e-6):
    assert isinstance(M, np.ndarray), '请将M矩阵转为numpy array格式'
    assert len(M.shape) == 2, '确保M矩阵为二维数组'
    assert (M >= 0).all(), '确保M矩阵每个元素都非负'

    n = M.shape[1]
    # 列归一化
    for i in range(n):
        sum_ = sum(M[:, i])
        assert sum_ != 0, '每列之和不可为0'
        M[:, i] /= sum_

    A = d * M + (1 - d) / n * np.full((n, n), 1)

    R_t1 = np.full((n, 1), 1 / n)
    for i in range(max_iter):
        R_t0 = R_t1
        R_t1 = np.matmul(A, R_t0)
        R_t1 = 1 / max(R_t1) * R_t1
        if sum(abs(R_t1 - R_t0)) < n * tol:
            break
    else:
        print(f'全部迭代后容差仍未为{sum(abs(R_t1 - R_t0))}')
    return R_t1 / sum(R_t1)


# 代数算法
def pagerank_algebra(M, d=0.85):
    assert isinstance(M, np.ndarray), '请将M矩阵转为numpy array格式'
    assert len(M.shape) == 2, '确保M矩阵为二维数组'
    assert (M >= 0).all(), '确保M矩阵每个元素都非负'

    n = M.shape[1]
    for i in range(n):
        sum_ = sum(M[:, i])
        assert sum_ != 0, '每列之和不可为0'
        M[:, i] /= sum_

    I = np.identity(3)
    inv = np.linalg.inv(I - d * M)
    one = np.full((3, 1), 1)
    R = np.matmul(inv, (1 - d) / n * one)
    return R


if __name__ == '__main__':
    sim_mat = np.array([[0., 1., 0.8],
                        [0.2, 0., 1.2],
                        [0.8, 6., 0.]])

    # networkx 求取pagerank值
    nx_graph = nx.from_numpy_array(sim_mat)
    print('networkx：\n', nx.pagerank(nx_graph, tol=1e-10))
    # 迭代法
    print('迭代法：\n', pagerank(sim_mat, tol=1e-10))
    # 迭代法
    print('幂法：\n', pagerank_power(sim_mat, tol=1e-10))
    # 迭代法
    print('代数法：\n', pagerank_algebra(sim_mat))
