#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 14:29
# @File    : ner.py
# @Software: PyCharm
# @System  : Windows
# @desc    : ner任务指标，如精确率、召回率、F1分数

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
    assert label and pred, 'label or pred is Null'
    assert len(label) == len(pred), 'label and pred have unequal length'
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
    assert label and pred, 'label or pred is Null'
    assert len(label) == len(pred), f'label and pred have unequal length, label: {len(label)}, pred: {len(pred)}'
    f1 = [0, 0, 0, 0, 0, 0]
    if isinstance(label[0], list):
        from itertools import chain
        label = list(chain.from_iterable(label))
        pred = list(chain.from_iterable(pred))

    correct_count = defaultdict(int)
    pred_count = defaultdict(int)
    label_count = defaultdict(int)

    last_pred, last_label = 'O', 'O'
    correct_chunk = None
    for i, (pred_single, label_single) in enumerate(zip(pred, label)):
        pred_start = pred_single.startswith('B')
        label_start = label_single.startswith('B')
        pred_entity = pred_single.split('-')[-1]
        label_entity = label_single.split('-')[-1]

        if correct_chunk:
            # 不存在前一个标签是'O'的情况
            if pred_single.split('-')[0] in ['B', 'O'] or last_pred.split('-')[-1] != correct_chunk:
                pred_end = True
            else:
                pred_end = False
            if label_single.split('-')[0] in ['B', 'O'] or last_label.split('-')[-1] != correct_chunk:
                label_end = True
            else:
                label_end = False

            if pred_end and label_end:
                correct_count[correct_chunk] += 1
                correct_chunk = None
            elif pred_end ^ label_end:
                correct_chunk = None

        if pred_start and label_start and pred_entity == label_entity:
            correct_chunk = label_entity
        if pred_start:
            pred_count[pred_entity] += 1
        if label_start:
            label_count[label_entity] += 1

        last_pred, last_label = pred_single, label_single

    if correct_chunk:
        correct_count[correct_chunk] += 1

    return_metric = cal_metrics(correct_count, pred_count, label_count,
                                sort_labels=sort_labels, digits=digits, return_avg_type=return_avg_type)

    return return_metric


def cal_metrics(correct_count, pred_count, label_count, sort_labels=None, digits=2, return_avg_type='macro'):
    '''
    得到统计的数据后计算各类别指标和平均指标，打印详细信息并返回指定平均类型的指标
    '''
    width = 12
    if not sort_labels:
        sort_labels = list(label_count.keys())
        for label in sort_labels:
            if label != 'O' and label.split('-')[0] not in ['B', 'I']:
                sort_labels.sort(key=lambda x: x)

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
    for k in sort_labels:
        f1[0] = correct_count[k] / pred_count[k] if pred_count[k] != 0 else 0
        f1[1] = correct_count[k] / label_count[k] if label_count[k] != 0 else 0
        f1[2] = 2 * f1[0] * f1[1] / (f1[0] + f1[1]) if f1[0] + f1[1] != 0 else 0
        f1[3:] = [correct_count[k], pred_count[k], label_count[k]]
        label_metric[k] = f1.copy()
        report += info.format(k, *f1, width=width, digits=digits)

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

    return_metric = []
    for i in avg_metric[return_avg_type][:3]:
        return_metric.append(round(i, digits))
    return return_metric


if __name__ == '__main__':
    y_pred = ['O', 'B-address', 'I-address', 'B-address', 'O', 'B-name', 'I-name', 'I-name']
    y_true = ['O', 'B-address', 'I-address', 'I-address', 'O', 'B-name', 'I-name', 'I-name']

    # f1 = get_fast_result_token_level(y_true, y_pred)
    # print('fast token micro f1 score', f1)

    # sort_labels = ['B-address', 'I-address', 'B-name', 'I-name']
    # f1 = get_result_token_level(y_true, y_pred, sort_labels=sort_labels, digits=3, return_avg_type='weighted')
    # print('token level f1 score', f1)

    sort_labels = ['address', 'name']
    f1 = get_result_entity_level(y_true, y_pred, sort_labels=sort_labels, digits=3, return_avg_type='macro')
    print('entity level f1 score', f1)