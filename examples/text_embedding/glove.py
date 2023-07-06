#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/6/22 21:50
# @File     : glove.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : glove词向量使用


# glove 预训练词向量下载地址：https://nlp.stanford.edu/projects/glove/
# 训练语料大多为英文，词表中只有几十个中文
import json

import numpy as np
from tqdm import tqdm

from milvus_func import MyMilvus
from milvus_func import milvus


# 把下载的txt文件转化成两个npy文件，方便后续提取词向量
# npy文件分别存储的词和向量，两个文件索引对应，成key-value形式
def txt_to_npy():
    embeddings_dict = {}
    with open("./data/glove/glove.6B.300d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            '''
                # 每一行第一个为词，后面为对应的向量，形式为：the 0.1512 0.1315 -0.1523 0.6215 -0.6859...
                # 其他的如 glove.840B.300d.zip 包中不止第一个是词，可能形式为to email 0.1512 0.1315 -0.1523 0.6215 -0.6859...
                # 可以使用如下代码分割
                
                word = ' '.join(values[:-300])
                vector = np.asarray(values[-300:], "float32")
            '''
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    np.save('./data/glove/wordsList', np.array(list(embeddings_dict.keys())))
    np.save('./data/glove/wordVectors', np.array(list(embeddings_dict.values()), dtype='float32'))


def create_milvus_db(collection_name):
    '''
    将glove中的文本和向量存储到milvus数据库库中
    :param collection_name:集合名
    :return:
    '''
    # 初始化自己包装的milvus类
    glove_milvus = MyMilvus(collection_name, 300)

    # 加载glove文本和向量
    wordsList = np.load('./data/glove/wordsList.npy')
    wordsList = wordsList.tolist()
    wordVectors = np.load('./data/glove/wordVectors.npy')

    # 分批导入milvus数据库中，一次不能加载太多向量，会超时导致加载失败
    batch_size = 10000
    bar = tqdm(range(0, len(wordVectors), batch_size))
    for i in bar:
        glove_milvus.insert_vectors(wordVectors[i: i+batch_size])
        bar.set_description(f'正在加载 {i//batch_size+1}/{len(wordVectors)//batch_size}')

    id2text = {id: wordsList[index] for id, index in glove_milvus.id2index.items()}
    '''
    导入milvus时会自动生成id，呈现1687595824404380005形式，需要将这种id与文本对应上
    即  "1687601053399834000": "the",
        "1687601053399834001": ",",
        "1687601053399834002": ".",
    '''
    with open(f'./data/glove/{collection_name}_id2text.json', 'w') as fp:
        fp.write(json.dumps(id2text, indent=2) + '\n')
    # with open(f'./data/glove/{collection_name}_id2index.json', 'w') as fp:
    #     fp.write(json.dumps(glove_milvus.id2index, indent=2) + '\n')


def search(collection_name, text):
    '''
    搜索glove词表中已有的相似词
    :param collection_name: 集合名
    :param text: str形式，输入一个glove词表中的词
    :return:
    '''
    wordsList = np.load('./data/glove/wordsList.npy')
    wordsList = wordsList.tolist()
    wordVectors = np.load('./data/glove/wordVectors.npy')

    # with open(f'./data/glove/{collection_name}_id2index.json', 'r') as fp:
    #     id2index = json.load(fp)
    with open(f'./data/glove/{collection_name}_id2text.json', 'r') as fp:
        id2text = json.load(fp)

    # 取出glove中改词的词向量，并拓展维度，形成shape:(1, 300)，满足搜索要求
    index = wordsList.index(text)
    text_vector = np.expand_dims(wordVectors[index], 0)

    status, results = milvus.search(collection_name=collection_name,
                                    query_records=text_vector,
                                    top_k=5,
                                    param={})
    # 根据相似词返回的id，从json文件中取出对应的文本
    # 由于glove词向量未归一化，所有距离得分大小变化很大
    text_score = []
    for r in results[0]:
        id, score = r.id, r.distance
        text = id2text.get(str(id))
        text_score.append([text, score])
        print(text, score)
    return text_score


if __name__ == '__main__':
    # txt_to_npy()

    # create_milvus_db('glove_300d')
    search('glove_300d', 'father')