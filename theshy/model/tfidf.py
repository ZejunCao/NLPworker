#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/6/27 21:49
# @File     : tfidf.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : tfidf模型得到关键词

# tfidf[word] = tf * idf = (该文档中word出现的次数 / 所有文档总词数) * (ln(总文档数 / 存在word的文档数) + 1)
# tf值是针对每个文档的，即每个文档中有几个word
# idf中+1为平滑项，防止idf值为0，有时在分子分母上再各加一，说防止分母为0，但是个人认为遍历的都是出现过的词，分母不会为0

from collections import defaultdict
import numpy as np


class Tfidf:
    def __init__(self, documents, use_idf=True, norm='l2'):
        self.documents = documents
        self.vocab = defaultdict(int)
        # 获取词表
        # defaultdict(<class 'int'>, {'查下': 0, '明天': 1, '天气': 2, '今天': 3})
        for text in documents:
            for t in text.split():
                self.vocab[t] = self.vocab.get(t, len(self.vocab))

        # 获得每个文档的tf向量
        # [[1. 1. 1. 0.]
        #  [1. 0. 1. 1.]]
        self.docment_vector = np.zeros((len(documents), len(self.vocab)))
        for i in range(len(documents)):
            for word in documents[i].split():
                self.docment_vector[i][self.vocab[word]] += 1

        if use_idf:
            self.idf = defaultdict(int)
            # 计算所有词汇的idf值, idf值在不同文档中是相等的
            self.idf_vector = np.zeros((len(self.vocab)))
            for k in self.vocab.keys():
                for document in documents:
                    if k in document.split():
                        self.idf[k] += 1
            # 先计算出现在文档集中的次数, 再取log
            # idf_vector: [1.         0.         1.         1.69314718]]
            for k, v in self.idf.items():
                self.idf[k] = np.log(len(documents) / v) + 1
                self.idf_vector[self.vocab[k]] = self.idf[k]
            # 得到最终的tf * idf
            self.docment_vector *= self.idf_vector

        if norm:
            # 这里只实现了l2标准化
            for i in range(len(self.docment_vector)):
                norm = np.sqrt(sum(self.docment_vector[i] ** 2))
                self.docment_vector[i] /= norm

    def search_similar(self, text):
        vec = np.zeros((len(self.vocab)))
        # 这里略过了词表之外的词
        for t in text.split():
            if t in self.vocab:
                vec[self.vocab[t]] += 1
        vec *= self.idf_vector
        vec /= np.sqrt(sum(vec ** 2))
        return vec


corpus = [
  "查下 明天 天气",
  "查下 今天 天气",
  # "帮我 查下 明天 北京 天气 怎么样",
  # "帮我 查下 今天 北京 天气 好不好",
  # "帮我 查询 去 北京 的 火车",
  # "帮我 查看 到 上海 的 火车",
  # "帮我 查看 特朗普 的 新闻",
  # "帮我 看看 有没有 北京 的 新闻",
  # "帮我 搜索 上海 有 什么 好玩的 上海",
  # "帮我 找找 上海 东方明珠 在哪"
]
target = '今天 天气 怎么样'
my_tfidf = Tfidf(corpus)
print('输出每个单词对应的 id 值：', my_tfidf.vocab)
print('返回idf值：', my_tfidf.idf)
print('返回各文档的向量：', my_tfidf.docment_vector)
print('指定文本的向量：', my_tfidf.search_similar(target))

# from sklearn.feature_extraction.text import TfidfVectorizer
# # 这里若设置use_idf=True, toarray()结果中是乘了idf的
# tfidf_vec = TfidfVectorizer(use_idf=True, smooth_idf=False, norm='l2')
#
# tfidf_matrix = tfidf_vec.fit_transform(corpus)
# # print('不重复的词:', tfidf_vec.get_feature_names())
# print('输出每个单词对应的 id 值:', tfidf_vec.vocabulary_)
# print('返回idf值:', tfidf_vec.idf_)
# # sklearn输出的tf值并没有除对应文档总词数，因为归一化后结果都是一样的
# print('返回tf值:', tfidf_matrix.toarray())
