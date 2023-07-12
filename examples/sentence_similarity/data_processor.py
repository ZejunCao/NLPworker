#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/1/7 23:33
# @File    : data_processor.py
# @Software: PyCharm
# @description: 数据处理，包括label_map处理，dataset建立

import json

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def data_process(path):
    # 读取每一条json数据放入列表中
    # 由于该json文件含多个数据，不能直接json.loads读取，需使用for循环逐条读取
    json_data = []
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            json_data.append(json.loads(line))

    return json_data


class Mydataset(Dataset):
    def __init__(self, config, file_path):
        self.config = config
        self.file_path = file_path
        self.tokenizer = config.tokenizer
        self.data = data_process(self.file_path)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        sentence1 = [b['sentence1'] for b in batch]
        sentence2 = [b['sentence2'] for b in batch]
        label = [b['label'] for b in batch]

        sentence1 = self.tokenizer(sentence1,
                                   pad_to_max_length=True,
                                   truncation=True,
                                   padding="longest",
                                   return_tensors="pt",
                                   )
        sentence2 = self.tokenizer(sentence2,
                                   pad_to_max_length=True,
                                   truncation=True,
                                   padding="longest",
                                   return_tensors="pt",
                                   )

        return {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'label': label,
        }


def Loader(config):
    if config.state == 'train':
        dataset = Mydataset(config, config.train_file_path)
        loader = DataLoader(dataset, batch_size=config.train_batch_size, num_workers=0, pin_memory=False, shuffle=True,
                                      collate_fn=dataset.collate_fn)
    elif config.state == 'eval':
        dataset = Mydataset(config, config.valid_file_path)
        loader = DataLoader(dataset, batch_size=config.valid_batch_size, num_workers=0, pin_memory=False, shuffle=False,
                                      collate_fn=dataset.collate_fn)
    elif config.state == 'pred':
        dataset = Mydataset(config, config.test_file_path)
        loader = DataLoader(dataset, batch_size=config.train_batch_size, num_workers=0, pin_memory=False, shuffle=False,
                                      collate_fn=dataset.collate_fn)
    else:
        raise Exception('请输入正确的config.state:["train", "eval", "pred"]')

    return loader
