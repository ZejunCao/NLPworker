#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/2/19 16:55
# @File    : crf.py
# @Software: PyCharm
# @System  : Windows
# @desc    : null

import torch
from torch import nn


# log sum exp 增强数值稳定性
# 改进了torch版本原始函数.可适用于两种情况计算得分
def log_sum_exp(vec):
    max_score, _ = torch.max(vec, dim=-1)
    max_score_broadcast = max_score.unsqueeze(-1).repeat_interleave(vec.shape[-1], dim=-1)
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1))


class CRF(nn.Module):
    def __init__(self, config, device='cpu'):
        super(CRF, self).__init__()
        self.label_map = config.label_map
        self.label_map_inv = config.label_map_inv
        self.tagset_size = len(self.label_map)
        self.device = device

        # 转移概率矩阵
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 增加开始和结束标志，并手动干预转移概率
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.transitions.data[self.label_map[self.START_TAG], :] = -10000
        self.transitions.data[:, self.label_map[self.STOP_TAG]] = -10000

    def _forward_alg(self, feats, seq_len):
        # 手动设置初始得分，让开始标志到其他标签的得分最高
        init_alphas = torch.full((self.tagset_size,), -10000.)
        init_alphas[self.label_map[self.START_TAG]] = 0.

        # 记录所有时间步的得分，为了解决序列长度不同问题，后面直接取各自长度索引的得分即可
        # shape：(batch_size, seq_len + 1, tagset_size)
        forward_var = torch.zeros(feats.shape[0], feats.shape[1] + 1, feats.shape[2], dtype=torch.float32,
                                  device=self.device)
        forward_var[:, 0, :] = init_alphas

        # 将转移概率矩阵复制 batch_size 次，批次内一起进行计算，矩阵计算优化，加快运行效率
        # shape：(batch_size, tagset_size) -> (batch_size, tagset_size, tagset_size)
        transitions = self.transitions.unsqueeze(0).repeat(feats.shape[0], 1, 1)
        # 对所有时间步进行遍历
        for seq_i in range(feats.shape[1]):
            # 取出当前词发射概率
            emit_score = feats[:, seq_i, :]
            # 前一时间步得分 + 转移概率 + 当前时间步发射概率
            tag_var = (
                    forward_var[:, seq_i, :].unsqueeze(1).repeat(1, feats.shape[2], 1)  # (batch_size, tagset_size, tagset_size)
                    + transitions
                    + emit_score.unsqueeze(2).repeat(1, 1, feats.shape[2])
            )
            # 这里必须调用clone，不能直接在forward_var上修改，否则在梯度回传时会报错
            cloned = forward_var.clone()
            cloned[:, seq_i + 1, :] = log_sum_exp(tag_var)
            forward_var = cloned

        # 按照不同序列长度不同取出最终得分
        forward_var = forward_var[range(feats.shape[0]), seq_len, :]
        # 手动干预,加上结束标志位的转移概率
        terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]].unsqueeze(0).repeat(feats.shape[0], 1)
        # 得到最终所有路径的分数和
        alpha = log_sum_exp(terminal_var)
        return alpha

    # 修改矩阵计算方式，加速计算
    def _score_sentence(self, feats, tags, seq_len):
        # 初始化,大小为(batch_size,)
        score = torch.zeros(feats.shape[0], device=self.device)
        # 将开始标签拼接到序列上起始位置，参与分数计算
        start = torch.tensor([self.label_map[self.START_TAG]], device=self.device).unsqueeze(0).repeat(feats.shape[0], 1)
        tags = torch.cat([start, tags], dim=1)
        # 在batch上遍历
        for batch_i in range(feats.shape[0]):
            # 采用矩阵计算方法，加快运行效率
            # 取出当前序列所有时间步的转移概率和发射概率进行相加，由于计算真实标签序列的得分，所以只选择标签的路径
            score[batch_i] = torch.sum(
                self.transitions[tags[batch_i, 1:seq_len[batch_i] + 1], tags[batch_i, :seq_len[batch_i]]]) \
                             + torch.sum(feats[batch_i, range(seq_len[batch_i]), tags[batch_i][1:seq_len[batch_i] + 1]])
            # 最后加上结束标志位的转移概率
            score[batch_i] += self.transitions[self.label_map[self.STOP_TAG], tags[batch_i][seq_len[batch_i]]]
        return score

    # 维特比算法得到最优路径,原始torch函数
    def viterbi_decode(self, feats):
        backpointers = []

        # 手动设置初始得分，让开始标志到其他标签的得分最高
        init_vvars = torch.full((1, self.tagset_size), -10000., device=self.device)
        init_vvars[0][self.label_map[self.START_TAG]] = 0

        # 用于记录前一时间步的分数
        forward_var = init_vvars
        # 传入的就是单个序列,在每个时间步上遍历
        for feat in feats:
            # 将上一时间步的总概率复制tagset_size次，以便一次性加上所有转移概率
            forward_var = forward_var.repeat(feat.shape[0], 1)
            next_tag_var = forward_var + self.transitions
            # 对每个标签位置取最大值的索引
            bptrs_t = torch.max(next_tag_var, 1)[1].tolist()
            # 取出当前时间步所有最大值的概率
            viterbivars_t = next_tag_var[range(forward_var.shape[0]), bptrs_t]
            # 加上当前时间步的发射概率
            forward_var = (viterbivars_t + feat).view(1, -1)
            # 记录最大值的索引，后续回溯用
            backpointers.append(bptrs_t)

        # 手动加入转移到结束标签的概率
        terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]]
        # 在最终位置得到最高分数所对应的索引
        best_tag_id = torch.max(terminal_var, 1)[1].item()
        # 最高分数
        path_score = terminal_var[0][best_tag_id]

        # 回溯，向后遍历得到最优路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # 弹出开始标签
        start = best_path.pop()
        assert start == self.label_map[self.START_TAG]  # Sanity check
        # 将路径反转
        best_path.reverse()
        return path_score, best_path

    def forward(self, feats, tags, seq_len):
        # 所有路径得分
        forward_score = self._forward_alg(feats, seq_len)
        # 标签路径得分
        gold_score = self._score_sentence(feats, tags, seq_len)
        # 返回 batch 分数的平均值
        return torch.mean(forward_score - gold_score)

