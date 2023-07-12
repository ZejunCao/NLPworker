#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/7/11 22:35
# @File    : rouge_.py
# @Software: PyCharm
# @System  : Windows
# @desc    : rouge指标计算



def sentence_level_rouge_n(list_reference, candidate, n=None):
    # 默认计算rouge 1-4
    if n is None:
        n = [1, 2, 3, 4]

    scores = {}
    for i in n:
        refer_count, cand_count, overlap_count = number_statistics(i, list_reference, candidate)
        scores.update(rouge_n(i, refer_count, cand_count, overlap_count))

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


if __name__ == '__main__':
    references1 = ['my', 'first', 'correct', 'sentence', 'sentence', 'penguin']
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
