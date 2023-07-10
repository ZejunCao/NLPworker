#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 19:40
# @File    : config.py
# @Software: PyCharm
# @System  : Windows
# @desc    : 配置文件base，设置配置默认值，保证examples未设置时不报错


class ConfigBase:
    def __init__(self):
        self.train_checkpoint = ''
        self.eval_checkpoint = ''
        self.pred_checkpoint = ''

        self.state = 'train'  # 'train', 'eval', 'pred'
