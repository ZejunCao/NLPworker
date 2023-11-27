#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/7/11 10:53
# @File     : bleu.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : bleu指标计算

import math
from collections import defaultdict


def sentence_level_bleu(list_reference, candidate, weights=None):
    '''
    计算句子级别的bleu指标，从bleu 1-4加权
    :param list_reference:参考集，列表形式 [['my', 'first'], ['correct', 'sentence']]
    :param candidate:候选集，预测文本 ['my', 'first', 'sentence']
    :param weights:不同的n-gram权重，自动归一化
    :return: bleu分数
    '''
    if not candidate:
        return 0
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]
    # 修改weights设置为负数的情况
    for w in weights:
        if w < 0:
            raise Exception(f"请设置weights为正数，当前{weights}")
    # 对于weights和不为1的情况，将其归一化
    if sum(weights) != 1:
        weights = [1 / sum(weights) * w for w in weights]

    # 统计参考集中n-gram数量
    reference_count = []
    for list_i in range(len(list_reference)):
        reference_count.append([defaultdict(int) for _ in range(4)])
        for i in range(len(list_reference[list_i])):
            for j in range(4):
                if i >= j:
                    reference_count[list_i][j][' '.join(list_reference[list_i][i-j: i+1])] += 1

    # 统计候选集中n-gram数量
    candidate_count = [defaultdict(int) for _ in range(4)]
    for i in range(len(candidate)):
        for j in range(4):
            if i >= j:
                candidate_count[j][' '.join(candidate[i-j: i+1])] += 1

    # 统计同时存在的n-gram的数量
    coexisting_count = [0, 0, 0, 0]
    for i in range(4):
        for k in candidate_count[i].keys():
            # 多个参考取词出现最大值
            rc = max([rc[i][k] for rc in reference_count])
            coexisting_count[i] += min(candidate_count[i][k], rc)

    # 计算长度惩罚系数
    refer_length = [len(l) for l in list_reference]
    refer_length.sort(key=lambda x: abs(x - len(candidate)))
    brevity_penalty = 1 - refer_length[0] / len(candidate) if len(candidate) < refer_length[0] else 0
    brevity_penalty = math.exp(brevity_penalty)

    # 取log加权
    score = 0
    for i in range(4):
        if sum(candidate_count[i].values()) > 0:
            p_i = coexisting_count[i] / sum(candidate_count[i].values())
            if p_i > 0:
                score += weights[i] * math.log(p_i)

    return math.exp(score) * brevity_penalty


if __name__ == '__main__':
    references = [['my', 'first', 'correct', 'sentence', 'penguin'], ['my', 'first', 'correct', 'sentence', 'sentence', 'sentence', 'crrect', 'sentence', 'sentence', 'sentence'], ['my', 'first', 'correct', 'sentence', 'sentence', 'crrect', 'crrect', 'crrect', 'crrect']]
    candidates = ['my', 'first', 'correct', 'sentence', 'sentence', 'apple', 'tiger', 'penguin']

    # nltk包测试样例，用于对比使用
    # from nltk.translate.bleu_score import corpus_bleu
    # score = corpus_bleu([references], [candidates], weights=(0.25, 0.25, 0.25, 0.25))
    # print(score)

    score = sentence_level_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
    print(score)