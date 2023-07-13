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
    def rouge_l_text(dp, candidate, text):
        m = min(len(dp), len(dp[0]))
        for i in range(m - 1):
            if dp[i+1][i+1] == dp[i][i] + 1 and dp[i+1][i] == dp[i][i] and dp[i][i+1] == dp[i][i]:
                text += [candidate[i]]
            else:
                begin = i
                break
        else:
            return text

        for j in range(begin+1, len(dp[0]) - 1):
            if dp[begin+1][j+1] == dp[begin][j] + 1 and dp[begin][j+1] == dp[begin][j] and dp[begin+1][j] == dp[begin][j]:
                t = rouge_l_text(dp[begin:, j:], candidate[begin:], text.copy())
                if len(t) == dp[-1][-1]: texts.append(t)

        for i in range(begin+1, len(dp) - 1):
            if dp[i+1][begin+1] == dp[i][begin] + 1 and dp[i][begin+1] == dp[i][begin] and dp[i+1][begin] == dp[i][begin]:
                t = rouge_l_text(dp[i:, begin:], candidate[i:], text.copy())
                if len(t) == dp[-1][-1]: texts.append(t)
        return text

    dp = [[0 for _ in range(len(reference) + 1)] for _ in range(len(candidate) + 1)]
    for i in range(1, len(candidate) + 1):
        for j in range(1, len(reference) + 1):
            if reference[j - 1] == candidate[i - 1]:
                dp[i][j] = max(dp[i - 1][j - 1] + 1, dp[i - 1][j], dp[i][j - 1])
            else:
                dp[i][j] = max(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    texts = []
    rouge_l_text(np.array(dp), candidate, [])
    common_len = 0
    for t in texts:
        common_len = max(common_len, len(set(t)))

    p = common_len / len(set(candidate))
    r = common_len / len(set(reference))
    return {
        f'rouge-l': {
            'precision': p,
            'recall': r,
            'f1': 2 * p * r / (p + r + 1e-8) if p + r != 0 else 0,
        }
    }






if __name__ == '__main__':
    references1 = ['my', 'first', 'correct', 'sentence', 'sentence', 'apple', 'tiger', 'penguin']
    candidates1 = ['my', 'first', 'correct', 'sentence', 'penguin']

    # references1 = ['a', 'b', 'c', 'd', 'e', 'f', 'd', 'g', 'h']
    # candidates1 = ['a', 'b', 'c', 'd', 'd', 'e', 'f', 'g']

    references = "my first correct sentence sentence apple tiger penguin"
    candidates = "my first correct sentence sentence penguin"

    # rouge包，pip install rouge安装
    from rouge import Rouge
    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    scores = rouge.get_scores(candidates, references)
    print(scores)

    scores = sentence_level_rouge_n(references1, candidates1, n=[1, 2])
    print(scores)
