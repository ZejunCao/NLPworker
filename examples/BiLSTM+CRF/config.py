#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 18:12
# @File    : config.py
# @Software: PyCharm
# @System  : Windows
# @desc    : 参数配置文件
import os


class Config():
    root_path = './data/cluener_public'
    train_file_path = os.path.join(root_path, 'train_json')
    valid_file_path = os.path.join(root_path, 'dev_json')

    embedding_size = 128
    hidden_dim = 768
    epochs = 50
    train_batch_size = 32
    valid_batch_size = 64
    lr = 0.001
    weight_decay = 1e-4