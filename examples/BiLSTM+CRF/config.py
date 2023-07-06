#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 18:12
# @File    : config.py
# @Software: PyCharm
# @System  : Windows
# @desc    : 参数配置文件
import os

import data_processor


class Config():
    root_path = './data/cluener_public'
    train_file_path = os.path.join(root_path, 'train.json')
    valid_file_path = os.path.join(root_path, 'dev.json')
    test_file_path = os.path.join(root_path, 'test.json')


    embedding_dim = 128
    hidden_dim = 768
    epochs = 50
    train_batch_size = 32
    valid_batch_size = 64
    lr = 0.001
    weight_decay = 1e-4

    label_map, label_map_inv = data_processor.get_label_map('./data/cluener_public/train.json')
    vocab, vocab_inv = data_processor.get_vocab('./data/cluener_public/train.json')
