#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/7/11 22:35
# @File    : rouge_.py
# @Software: PyCharm
# @System  : Windows
# @desc    : rouge指标计算

import numpy as np

def sentence_level_rouge_n(list_reference, candidate, n=None):
    # 默认计算rouge 1-4
    if n is None:
        n = [1, 2, 3, 4]

    scores = {}
    for i in n:
        refer_count, cand_count, overlap_count = number_statistics(i, list_reference, candidate)
        scores.update(rouge_n(i, refer_count, cand_count, overlap_count))

    scores.update(rouge_l(list_reference, candidate))
    return scores


def rouge_n(n, refer_count, cand_count, overlap_count):
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
    dp = [[0 for _ in range(len(reference) + 1)] for _ in range(len(candidate) + 1)]
    for i in range(1, len(candidate) + 1):
        for j in range(1, len(reference) + 1):
            if reference[j - 1] == candidate[i - 1]:
                dp[i][j] = max(dp[i - 1][j - 1] + 1, dp[i - 1][j], dp[i][j - 1])
            else:
                dp[i][j] = max(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    rouge_l_text(np.array(dp), [], candidate)
    p = dp[-1][-1] / len(set(candidate))
    r = dp[-1][-1] / len(set(reference))
    return {
        f'rouge-l': {
            'precision': p,
            'recall': r,
            'f1': 2 * p * r / (p + r + 1e-8) if p + r != 0 else 0,
        }
    }


def rouge_l_text(dp, text, candidate):
    m = min(len(dp), len(dp[0]))
    for i in range(m - 1):
        if dp[i+1][i+1] == dp[i][i] + 1:
            text.append(candidate[i])
        else:
            begin = i
            break
    else:
        return

    for j in range(begin+1, len(dp[0]) - 1):
        if dp[begin+1][j+1] == dp[begin][j] + 1:
            rouge_l_text(dp[begin:, j:], text, candidate[j-1:])

    for i in range(begin+1, len(dp) - 1):
        if dp[i+1][begin+1] == dp[i][begin] + 1:
            rouge_l_text(dp[i:][begin:], text, candidate[begin:])





if __name__ == '__main__':
    references1 = ['my', 'first', 'correct', 'sentence', 'apple', 'tiger', 'sentence', 'penguin', 'a']
    candidates1 = ['my', 'first', 'correct', 'sentence', 'sentence', 'apple', 'tiger', 'penguin']

    candidates = "my first correct sentence sentence penguin"
    references = "my first correct sentence sentence apple tiger penguin"

    # rouge包，pip install rouge安装
    from rouge import Rouge
    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-3", "rouge-4", "rouge-l"])
    scores = rouge.get_scores(candidates, references)
    print(scores)

    scores = sentence_level_rouge_n(references1, candidates1)
    print(scores)
