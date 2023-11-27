#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 19:40
# @File    : config.py
# @Software: PyCharm
# @System  : Windows
# @desc    : 配置文件base，设置配置默认值，保证examples未设置时不报错

import os
import logging
import datetime

class ConfigBase:
    def __init__(self):
        self.train_checkpoint = ''
        self.eval_checkpoint = ''
        self.pred_checkpoint = ''

        self.state = 'train'  # 'train', 'eval', 'pred'
        self.project_path = os.getcwd()
        # os.makedirs(os.path.join(self.project_path, 'checkpoint'), exist_ok=True)

        # 创建保存checkpoint文件夹
        self.checkpoint_path = os.path.join(self.project_path, 'checkpoint', f"{datetime.datetime.now().strftime('%m%d-%H-%M-%S')}")
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.logger = logging.getLogger(name='nlpworker')
        self.logger.setLevel(logging.DEBUG)

        # 1、创建一个handler，该handler往console打印输出
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(filename=f'{self.checkpoint_path}/info.log')
        standard_formatter = logging.Formatter('%(asctime)s %(filename)s line:%(lineno)d %(levelname)s %(message)s %(process)d')

        console_handler.setFormatter(standard_formatter)
        file_handler.setFormatter(standard_formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)