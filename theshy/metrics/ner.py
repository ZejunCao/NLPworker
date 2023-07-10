#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 14:29
# @File    : ner.py
# @Software: PyCharm
# @System  : Windows
# @desc    : ner任务指标，如精确率、召回率、F1分数
import time

import numpy as np
from collections import defaultdict


def get_fast_result_token_level(label, pred):
    '''
    token级别快速指标获取，micro平均指标，不打印，只返回指标结果
    :param label: 标签序列，输入可为单列表或嵌套列表
    :param pred: 预测序列，输入可为单列表或嵌套列表
    :return: 返回'micro'类型的精确率、召回率、f1-score，以列表形式，如[0.45, 0.43, 0.44]
    '''
    assert label and pred, 'label or pred is Null'
    assert len(label) == len(pred), 'label and pred have unequal length'
    if isinstance(label[0], list):
        from itertools import chain
        label = list(chain.from_iterable(label))
        pred = list(chain.from_iterable(pred))
    label = np.array(label)
    pred = np.array(pred)

    correct_num = np.bitwise_and(label != 'O', label == pred).sum()
    target_num = (label != 'O').sum()
    pred_num = (pred != 'O').sum()
    f1 = [0, 0, 0]
    f1[0] = round(correct_num / pred_num, 2) if pred_num != 0 else 0
    f1[1] = round(correct_num / target_num, 2) if target_num != 0 else 0
    f1[2] = round(2 * f1[0] * f1[1] /(f1[0] + f1[1]), 2) if f1[0] + f1[1] != 0 else 0
    return f1


def get_result_token_level(label, pred, sort_labels=None, digits=2, return_avg_type='macro'):
    '''
    token级别指标，打印具体信息并返回总体指标
    :param label: 标签序列，输入可为单列表或嵌套列表
    :param pred: 预测序列，输入可为单列表或嵌套列表
    :param sort_labels -> list: 标签类别，决定打印顺序；若有则按给定顺序，若无则按标签序列中的出现顺序
    :param digits: 保留小数位数，同时控制打印位数和返回结果位数
    :param return_avg_type: 返回多类别平均值类型；可选'micro', 'macro', 'weighted'
    :return: 返回总体指定类型的精确率、召回率、f1-score，以列表形式，如[0.45, 0.43, 0.44]
    '''
    # 仅支持BIO格式，标签格式为 'B-scene', 'I-scene', 'O'
    # 确保输入序列不为空
    assert label and pred, 'label or pred is Null'
    # 确保输入序列长度相等
    assert len(label) == len(pred), 'label and pred have unequal length'

    # 若输入序列是嵌套列表形式,则将其展开成一个一维列表
    if isinstance(label[0], list):
        from itertools import chain
        label = list(chain.from_iterable(label))
        pred = list(chain.from_iterable(pred))

    correct_count = defaultdict(int)
    pred_count = defaultdict(int)
    label_count = defaultdict(int)

    for pred_single, label_single in zip(pred, label):
        if pred_single == label_single:
            correct_count[pred_single] += 1
        pred_count[pred_single] += 1
        label_count[label_single] += 1

    return_metric = cal_metrics(correct_count, pred_count, label_count,
                                sort_labels=sort_labels, digits=digits, return_avg_type=return_avg_type)

    return return_metric


def get_result_entity_level(label, pred, sort_labels=None, digits=2, return_avg_type='macro'):
    '''
    实体级别指标，打印具体信息并返回总体指标
    :param label: 标签序列，输入可为单列表或嵌套列表
    :param pred: 预测序列，输入可为单列表或嵌套列表
    :param sort_labels -> list: 标签类别，决定打印顺序；若有则按给定顺序，若无则按标签序列中的出现顺序
    :param digits: 保留小数位数，同时控制打印位数和返回结果位数
    :param return_avg_type: 返回多类别平均值类型；可选'micro', 'macro', 'weighted'
    :return: 返回总体指定类型的精确率、召回率、f1-score，以列表形式，如[0.45, 0.43, 0.44]
    '''
    # 仅支持BIO格式，标签格式为 'B-scene', 'I-scene', 'O'
    # 确保输入序列不为空
    assert label and pred, 'label or pred is Null'
    # 确保输入序列长度相等
    assert len(label) == len(pred), f'label and pred have unequal length, label: {len(label)}, pred: {len(pred)}'

    # 若输入序列是嵌套列表形式,则将其展开成一个一维列表
    if isinstance(label[0], list):
        from itertools import chain
        label = list(chain.from_iterable(label))
        pred = list(chain.from_iterable(pred))

    # 初始化记录类别个数的字典,分别记录各个类别的正确个数、预测个数、真实值个数
    correct_count = defaultdict(int)
    pred_count = defaultdict(int)
    label_count = defaultdict(int)

    correct_chunk = None
    for i, (pred_single, label_single) in enumerate(zip(pred, label)):
        # 判断是否为B开头的类别
        pred_start = pred_single.startswith('B')
        label_start = label_single.startswith('B')
        # 取出标签的类别
        pred_entity = pred_single.split('-')[-1]
        label_entity = label_single.split('-')[-1]

        if correct_chunk:
            # 如果进入当前条件，不存在前一个标签是'O'的情况
            # 当前为'B'或'O'，或者当前实体类别与之前记录不同，则视为该实体结束，下面真实标签同理
            if pred_single.split('-')[0] in ['B', 'O'] or pred_single.split('-')[-1] != correct_chunk:
                pred_end = True
            else:
                pred_end = False
            if label_single.split('-')[0] in ['B', 'O'] or label_single.split('-')[-1] != correct_chunk:
                label_end = True
            else:
                label_end = False

            # 必须预测标签和真实标签结束位置相同才视为预测正确，正确字典该类别加1，并清空correct_chunk
            if pred_end and label_end:
                correct_count[correct_chunk] += 1
                correct_chunk = None
            elif pred_end ^ label_end:
                correct_chunk = None

        # 如果当前预测标签和真实标签都是B开头，并且类别相同，则记录下这个类别，并在下一次循环中判断这整个实体是否预测正确
        if pred_start and label_start and pred_entity == label_entity:
            correct_chunk = label_entity
        # 只要预测标签是B开头，则预测字典中该类别个数加1，下面真实值字典同理
        if pred_start:
            pred_count[pred_entity] += 1
        if label_start:
            label_count[label_entity] += 1

    # 结尾位置需特殊处理，若前面有一个实体未结束，这里加上
    if correct_chunk:
        correct_count[correct_chunk] += 1

    # 得到各类被的正确个数、预测个数、真实值个数字典后，计算precision、recall、f1-score指标，这个与token级别计算方式相同，所以单独列个函数
    return_metric = cal_metrics(correct_count, pred_count, label_count,
                                sort_labels=sort_labels, digits=digits, return_avg_type=return_avg_type)

    return return_metric


def cal_metrics(correct_count, pred_count, label_count, sort_labels=None, digits=2, return_avg_type='macro'):
    '''
    得到统计的数据后计算各类别指标和平均指标，打印详细信息并返回指定平均类型的指标
    '''
    width = 12
    if not sort_labels:
        # 如果没有给定类别列表，则取预测个数字典和真实标签字典中的类别，并按首字母排序
        sort_labels = list(set(pred_count.keys()) | set(label_count.keys()))
        if len(sort_labels) > 1 and sort_labels[0].split('-')[0] not in ['B', 'I', 'O']:
            sort_labels.sort()

    # 设置打印宽度，取决于类别字体长度
    for k in sort_labels:
        width = max(width, len(k))

    report = ''
    headers = ['precision', 'recall', 'f1-score', 'correct_num', 'pred_num', 'label_num']
    info = '{:>{width}s} ' + ' {:>9}' * len(headers) + '\n\n'
    report += info.format('', *headers, width=width)

    label_metric = defaultdict(list)
    avg_metric = defaultdict(list)
    info = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}' * 3 + '\n'
    f1 = [0, 0, 0, 0, 0, 0]
    # 分别打印各个类别的precision、recall、f1-score、正确个数、预测个数、真实标签个数
    for k in sort_labels:
        f1[0] = correct_count[k] / pred_count[k] if pred_count[k] != 0 else 0
        f1[1] = correct_count[k] / label_count[k] if label_count[k] != 0 else 0
        f1[2] = 2 * f1[0] * f1[1] / (f1[0] + f1[1]) if f1[0] + f1[1] != 0 else 0
        f1[3:] = [correct_count[k], pred_count[k], label_count[k]]
        label_metric[k] = f1.copy()
        report += info.format(k, *f1, width=width, digits=digits)

    # 最后打印'micro', 'macro', 'weighted'多类别指标
    avg = ['micro', 'macro', 'weighted']
    if 'O' in correct_count.keys():
        correct_count.pop('O')
    if 'O' in pred_count.keys():
        pred_count.pop('O')
    if 'O' in label_count.keys():
        label_count.pop('O')
    if 'O' in label_metric.keys():
        label_metric.pop('O')

    report += '\n'
    f1[3:] = [sum(correct_count.values()), sum(pred_count.values()), sum(label_count.values())]
    for a in avg:
        if a == 'micro':
            f1[0] = sum(correct_count.values()) / sum(pred_count.values()) if sum(pred_count.values()) != 0 else 0
            f1[1] = sum(correct_count.values()) / sum(label_count.values()) if sum(label_count.values()) != 0 else 0
            f1[2] = 2 * f1[0] * f1[1] / (f1[0] + f1[1]) if f1[0] + f1[1] != 0 else 0
        elif a == 'macro':
            for i in range(3):
                f1[i] = np.nanmean([l[i] for l in label_metric.values()])
                f1[i] = 0 if np.isnan(f1[i]) else f1[i]
        elif a == 'weighted':
            f1[0] = sum([i[0] * i[5] / f1[5] for i in label_metric.values() if f1[5] != 0])
            f1[1] = sum([i[1] * i[5] / f1[5] for i in label_metric.values() if f1[5] != 0])
            f1[2] = sum([i[2] * i[5] / f1[5] for i in label_metric.values() if f1[5] != 0])

        avg_metric[a] = f1.copy()
        report += info.format(a + ' avg', *f1, width=width, digits=digits)
    print(report)

    return_metric = {}
    # 将最终指标precision、recall、f1-score三个值返回，用于模型训练评估
    # 以字典形式返回
    return_metric['precision'] = round(avg_metric[return_avg_type][0], digits)
    return_metric['recall'] = round(avg_metric[return_avg_type][1], digits)
    return_metric['f1'] = round(avg_metric[return_avg_type][2], digits)
    return return_metric


# 预测序列实体提取
def chunks_extract(pred):
    if not pred:
        return []

    cur_entity = None
    res = []
    st_idx, end_idx = 0, 0
    for i, pred_single in enumerate(pred):
        pred_start_B = pred_single.startswith('B')
        pred_entity = pred_single.split('-')[-1]

        if cur_entity:
            if pred_start_B or cur_entity != pred_entity:
                res.append({
                    'st_idx': st_idx,
                    'end_idx': i,
                    'label': cur_entity
                })
                cur_entity = None
        if pred_start_B:
            st_idx = i
            cur_entity = pred_entity
    if cur_entity:
        res.append({
            'st_idx': st_idx,
            'end_idx': len(pred),
            'label': cur_entity,
        })
    return res

if __name__ == '__main__':
    y_pred = ['O', 'B-address', 'I-address', 'I-name', 'O', 'B-name', 'I-name', 'I-name']
    y_true = ['O', 'B-address', 'I-address', 'I-address', 'O', 'B-name', 'I-name', 'I-name']

    # 几种特殊情况，主要看预测实体的个数，不必在意指标
    # y_pred = ['B-address', 'I-address', 'I-name']  # 两个类别的I挨着
    # y_pred = ['B-address', 'B-name', 'I-name']  # 两个类别的B挨着
    # y_pred = ['B-address', 'B-address', 'O']  # 相同类别的B挨着
    # y_pred = ['B-address', 'O', 'O']  # B后面是O
    # y_pred = ['O', 'I-address', 'O']  # O后面是I

    # y_true = ['B-address', 'I-address', 'I-name']
    # f1 = get_fast_result_token_level(y_true, y_pred)
    # print('fast token micro f1 score', f1)

    # sort_labels = ['B-address', 'I-address', 'B-name', 'I-name']
    # f1 = get_result_token_level(y_true, y_pred, sort_labels=sort_labels, digits=3, return_avg_type='weighted')
    # print('token level f1 score', f1)

    sort_labels = ['address', 'name']
    f1 = get_result_token_level(y_true, y_pred, sort_labels=sort_labels, digits=3, return_avg_type='macro')
    print('entity level f1 score', f1)