#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/8/20 13:50
# @File     : similarity.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : 各种相似度计算方法，包括余弦距离、皮尔逊相关系数、杰卡德相似度

import numpy as np

def pearson(x, y):
    '''
    pearson相关系数有四种计算公式，这里使用
    p_{x, y} = (N∑xy - ∑x∑y) / (sqrt(N∑x^2 - (∑x)^2) * sqrt(N∑y^2 - (∑y)^2))
    :param x1:
    :param x2:
    :return:
    '''
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    a = n * sum(x * y) - sum(x) * sum(y)
    b = np.sqrt(n * sum(x ** 2) - sum(x) ** 2) * np.sqrt(n * sum(y ** 2) - sum(y) ** 2)
    return a / b

if __name__ == '__main__':
    x = np.array([1, 3, 4, 3, 1])
    y = np.array([2, 1, 3, 3, 5])
    pc = pearson(x, y)
    print(pc)