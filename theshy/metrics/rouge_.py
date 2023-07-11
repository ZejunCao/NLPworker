#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/7/11 22:35
# @File    : rouge_.py
# @Software: PyCharm
# @System  : Windows
# @desc    : rouge指标计算

# rouge包，pip install rouge安装
from rouge import Rouge

def sentence_level_rouge_n(list_reference, candidate):
    scores = {"rouge-1": {}}
    # 对于重复文本只看做一次，所以直接取set
    candidate_set = set(candidate)
    list_reference_set = set(list_reference)

    overlap = sum([1 if i in list_reference_set else 0 for i in candidate_set])
    p, r = overlap / len(candidate), overlap / len(list_reference_set)
    scores['rouge-1']['precision'] = p
    scores['rouge-1']['recall'] = r
    scores['rouge-1']['f1'] = 2 * p * r / (p + r) if p + r != 0 else 0


    return


references1 = ['my', 'first', 'correct', 'sentence', 'sentence', 'penguin']
candidates1 = ['my', 'first', 'correct', 'sentence', 'sentence', 'apple', 'tiger', 'penguin']

candidates = "my first correct sentence penguin"
references = "my first correct sentence apple tiger penguin"

rouge = Rouge()
scores = rouge.get_scores(candidates, references)
print(scores)
scores = sentence_level_rouge_n(references1, candidates1)
print(scores)
