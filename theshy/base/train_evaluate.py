#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author      : Cao Zejun
# @Time        : 2023/12/10 23:16
# @File        : train_evaluate.py
# @Software    : Pycharm
# @description :

class TrainEvaluateBase:
    model = None
    config = None
    def __init__(self, model=None):
        if model:
            TrainEvaluateBase.config = model.config
            TrainEvaluateBase.model = model.to(model.config.device)
