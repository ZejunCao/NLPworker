#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 19:40
# @File    : config.py
# @Software: PyCharm
# @System  : Windows
# @desc    : 配置文件base，设置配置默认值，保证examples未设置时不报错

import os
import sys
import datetime
import time

import pytz
from loguru import logger


class ConfigBase:
    def __init__(self):
        self.train_checkpoint = ''
        self.eval_checkpoint = ''
        self.pred_checkpoint = ''

        self.state = 'train'  # 'train', 'eval', 'pred'
        self.eval_pre_train = False  # 设置是否需要在开始训练前验证一次
        self.checkpoint_num = 2  # 保留权重的数量，默认2个，代表最近和历史指标最优的权重
        self.log_per_step = 1  # 设置每隔多少步打印一次，默认一步一打印

        self.project_path = os.getcwd()
        self.checkpoint_path = os.path.join(self.project_path, 'checkpoint')
        self.log_path = os.path.join(self.project_path, 'log')
        self.start_run_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%m%d_%H_%M_%S')
        self.checkpoint_cur_path = os.path.join(self.checkpoint_path, self.start_run_time)

        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        def format_message(record):
            # 设置北京时间，由于历史原因，这里使用'Asia/Shanghai'代表
            formatted_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
            record = f'<green>{formatted_time}</green>' \
                     ' | <level>{level: <8}</level> ' \
                     '| <magenta>{process}</magenta>:<yellow>{thread}</yellow> ' \
                     '| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<yellow>{line}</yellow> - <level>{message}</level>'
            return record

        logger.remove(0)
        logger.add(sys.stderr, format=format_message, level="DEBUG")
        logger.add(f"{self.log_path}/{self.start_run_time}.log")