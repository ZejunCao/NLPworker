#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/7/11 22:35
# @File    : rouge_.py
# @Software: PyCharm
# @System  : Windows
# @desc    : rouge指标计算

# 这里只计算单个参考文本和单个候选文本之间的rouge分数
# 当出现多个时，分别计算互相之间的rouge分数，然后取最大值，由于rouge指标注重的是召回率，所以推荐取以召回率为基准取最大值

import time
import numpy as np


def sentence_level_rouge_n(reference, candidate, n=None):
    # 默认计算rouge 1-4
    if n is None:
        n = [1, 2, 3, 4]

    scores = {}
    for i in n:
        refer_count, cand_count, overlap_count = number_statistics(i, reference, candidate)
        scores.update(compute_metric(i, refer_count, cand_count, overlap_count))

    scores.update(rouge_l(reference, candidate))
    return scores


def compute_metric(n, refer_count, cand_count, overlap_count):
    p, r = overlap_count / cand_count, overlap_count / refer_count
    return {
        f'rouge-{n}': {
            'precision': p,
            'recall': r,
            'f1': 2 * p * r / (p + r + 1e-8) if p + r != 0 else 0,
        }
    }


def number_statistics(n, reference, candidate):
    refer = []
    for i in range(n, len(reference)+1):
        refer.append(' '.join(reference[i-n: i]))
    # 对于重复文本只看做一次，所以直接取set
    refer = set(refer)

    cand = []
    for i in range(n, len(candidate)+1):
        cand.append(' '.join(candidate[i-n: i]))
    cand = set(cand)

    overlap = sum([1 if i in refer else 0 for i in cand])

    return len(refer), len(cand), overlap


def rouge_l(reference, candidate):
    def rouge_l_text(dp, text):
        i = len(dp) - 1
        j = len(dp[0]) - 1
        while i > 0 and j > 0:
            if dp[i-1][j] < dp[i][j] and dp[i][j-1] < dp[i][j]:
                text.append(candidate[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] == dp[i][j] and dp[i][j-1] == dp[i][j]:
                rouge_l_text(dp[:i+1, :j], text.copy())
                rouge_l_text(dp[:i, :j+1], text.copy())
                break
            elif dp[i-1][j] == dp[i][j]:
                i -= 1
            elif dp[i][j-1] == dp[i][j]:
                j -= 1
        else:
            texts.append(text[::-1])

    dp = [[0 for _ in range(len(reference) + 1)] for _ in range(len(candidate) + 1)]
    for i in range(1, len(candidate) + 1):
        for j in range(1, len(reference) + 1):
            if reference[j - 1] == candidate[i - 1]:
                dp[i][j] = max(dp[i - 1][j - 1] + 1, dp[i - 1][j], dp[i][j - 1])
            else:
                dp[i][j] = max(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    texts = []
    rouge_l_text(np.array(dp), [])
    # 将多个最长子序列合并，然后再取set
    common_len = len(set([t for text in texts for t in text]))

    return compute_metric('l', len(set(reference)), len(set(candidate)), common_len)


if __name__ == '__main__':
    references1 = ['a', 'b', 'c', 'd', 'e', 'f', 'd', 'e', 'g']
    candidates1 = ['a', 'b', 'c', 'd', 'd', 'e', 'f', 'g']

    references = "a b c d e f d e g"
    candidates = "a b c d d e f g"
    '''
    先取最长子序列，然后再取set
    references = "a a a b"
    candidates = "a b a a d e f g"
    '''
    # rouge包，pip install rouge安装
    from rouge import Rouge
    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    a = time.time()
    for i in range(50000):
        scores = rouge.get_scores(candidates, references)
    print(time.time()-a)
    print(scores)

    a = time.time()
    for i in range(50000):
        scores = sentence_level_rouge_n(references1, candidates1, n=[1, 2])
    print(time.time()-a)
    print(scores)
