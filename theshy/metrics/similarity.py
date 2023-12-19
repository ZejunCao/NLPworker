#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/8/20 13:50
# @File     : similarity.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : 各种相似度计算方法，包括余弦距离、皮尔逊相关系数、杰卡德相似度

import numpy as np
from loguru import logger

def pearson_cal(x, y):
    '''
    pearson相关系数有四种计算公式，这里使用
    p_{x, y} = (N∑xy - ∑x∑y) / (sqrt(N∑x^2 - (∑x)^2) * sqrt(N∑y^2 - (∑y)^2))
    :param x1:
    :param x2:
    :return:
    '''
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    # x = np.array(x)
    # y = np.array(y)
    n = len(x)

    a = n * sum(x * y) - sum(x) * sum(y)
    b = np.sqrt(n * sum(x ** 2) - sum(x) ** 2) * np.sqrt(n * sum(y ** 2) - sum(y) ** 2)
    res = a / b
    logger.info(f"Pearson coefficient: {res}")
    return res

def accurary_cal(label, pred):
    if isinstance(label, list):
        label = np.array(label)
    if isinstance(pred, list):
        pred = np.array(pred)

    if len(pred.shape) != 1:
        pred = np.argmax(pred, axis=-1)
    dic = {'accurary': sum(pred == label) / pred.shape[0]}
    logger.info(f"accurary: {dic['accurary']}")
    return dic

# TODO：对于数据数值相同时如何计算排名
def spearman_cal(x, y):
    '''
    spearman相关系数公式：p_{x, y} = 1 - 6 * ∑d_i^2 / {n * (n^2 - 1)}
    d_i = rank_x[i] - rank_y[i]
    :param x:
    :param y:
    :return:
    '''
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    rank_x = x.copy()
    order_index = np.argsort(x)[::-1]
    equal_num = 0
    for rank_i, i in enumerate(range(len(order_index)), start=1):
        if i > 0 and x[order_index[i]] == x[order_index[i-1]]:
            equal_num += 1
        else:
            equal_num = 0
        rank_x[order_index[i]] = rank_i - equal_num

    rank_y = y.copy()
    order_index = np.argsort(y)[::-1]
    equal_num = 0
    for rank_i, i in enumerate(range(len(order_index)), start=1):
        if i > 0 and y[order_index[i]] == y[order_index[i-1]]:
            equal_num += 1
        else:
            equal_num = 0
        rank_y[order_index[i]] = rank_i - equal_num

    d = sum([(rank_x[i] - rank_y[i]) ** 2 for i in range(len(rank_x))])
    res = 1 - 6 * d / (len(rank_x) * (len(rank_x) ** 2 - 1))
    logger.info(f"Spearman coefficient: {res}")
    return res

def spearmanr_cal1(x, y):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)

    d = sum([(x[i] - y[i]) ** 2 for i in range(len(x))])
    res = 1 - 6 * d / (len(x) * (len(x) ** 2 - 1))
    print(res)
    return res


if __name__ == '__main__':
    # x = np.array([1, 3, 4, 3, 1])
    # y = np.array([2, 1, 3, 3, 5])
    # pc = pearson(x, y)
    # accurary_cal(x, y)
    # print(pc)

    import numpy as np
    from scipy import stats

    # res1 = stats.spearmanr([1, 2, 3, 3, 4], [1, 2, 3, 4, 5])
    # print(res1.statistic)
    # res2 = spearman_cal([35, 23, 47, 17, 10, 43, 9, 6, 28], [30, 33, 45, 23, 8, 49, 12,  4, 31])
    # print(res2)

    # res1 = stats.spearmanr([0, 0, 0, 1, 1, 1], [1, 2, 3, 4, 5, 6])
    # # res1 = stats.spearmanr([1, 2, 3, 3, 3, 4], [1, 2, 3, 4, 5, 6])
    # print(res1)
    # res2 = spearman_cal([1, 2, 3, 3, 4], [1, 2, 3, 4, 5])
    res2 = spearmanr_cal1([1,2,3,4,5,6,7], [2.5,2.5,2.5,2.5,6,6,6])
