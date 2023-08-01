#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/7/29 11:30
# @File     : decision_tree.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : 决策树实现

'''
    ID3：信息增益，离散值分割
    C4.5：信息增益比，离散值与连续值分割，可手动指定
    测试集分类：对于未出现值，使用当前节点标签

'''

# TODO CART决策树

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


class TreeNode:
    def __init__(self, feature_index=-1, label=-1):
        self.feature_index = feature_index
        self.feature_continuous = False
        self.split_value = 0
        self.label = label
        self.next = {}


class Decision_Tree:
    def __init__(self):
        pass

    def fit(self, input, label, method='ID3', continuous=[]):
        '''

        :param input:
        :param label:
        :param method: 决策树训练方法，指定'ID3', 'C4.5', 'CART'其中一种
        :param continuous->List: 指定哪些特征是连续值，默认全部视为离散值，设置格式为[0,3]，代表第0和第3个特征是连续值
        :return:
        '''
        assert method in ['ID3', 'C4.5', 'CART'], "请输入'ID3', 'C4.5', 'CART'其中一种"
        input = self.to_numpy(input)
        label = self.to_numpy(label)
        assert not continuous or max(continuous) < input.shape[-1], f'连续值索引范围请设置0-{input.shape[-1]-1}'

        self.n_target = len(set(label))
        self.feature_num = input.shape[-1]
        if method == 'ID3':
            self.root = self.ID3(input, label)
        elif method == 'C4.5':
            # self.root = self.C4_5(input, label)
            self.root = self.C4_5_continuous(input, label, continuous=continuous)

    # 递归创建树
    def ID3(self, input, label, already_index=[]):
        '''
        使用信息增益来选择特征
        D代表整个数据集，特征分别为A_1、A_2、、、A_n，假设第i个特征的所有互异取值为x^i_1、x^i_2、x^i_3、、、x^i_n
        H(D) = -∑(p_i * log2(p_i))   这里的p_i代表第i个标签的统计概率,统计概率就是 i的个数/总个数

        信息增益：g(D, A_i) = H(D) - H(D|A_i) = H(D) - ∑(|x^i_j| / |D| * H(D_j))
        i代表第i个特征，j代表第i个特征的第j种取值，共有n种取值，|x^i_j|代表第i个特征的第j个取值个数，|D|代表数据集D的个数

        H(D_j) = -∑(p_i * log2(p_i))   这里的p_i = x^i_j中标签为k的个数 / x^i_j的个数  k=(0,1,2...)
        :param input: 输入训练数据，不会出现空的情况
        :param label:
        :param already_index: 记录已经进行过决策的特征
        :return: node
        '''
        node = TreeNode()
        cur_n_target = len(set(label))
        # 将当前节点的标签值设为所含样本最多的类别，不管子节点还是父节点，防止出现训练数据之外的特征值
        label_num = [sum(label == i) for i in range(self.n_target)]
        node.label = label_num.index(max(label_num))
        # 如果当前数据集全部是一样的标签
        if cur_n_target == 1:
            node.label = label[0]
            return node
        # 如果遍历过所有特征了
        if len(already_index) == self.feature_num:
            return node

        # 计算 H(D) = -∑(p_i * log2(p_i))
        H_D = [sum(label == c) / len(label) for c in set(label)]  # p_i
        H_D = sum([-p * np.log2(p) for p in H_D])  # -∑(p_i * log2(p_i))
        # 用来存储各个特征的信息增益
        g_D_A = []
        for i in range(input.shape[-1]):
            # 已经遍历过的特征无需再次计算
            if i in already_index:
                g_D_A.append(0)
                continue
            # 统计当前特征A_i各个取值的个数
            counter = Counter(input[:, i])
            # ∑(|x^i_j| / |D| * H(D_j)), 具体i,j,k字母有变化
            H_D_Ai = 0
            for k, v in counter.items():
                H_D_j = [sum(label[input[:, i] == k] == j) / v for j in range(cur_n_target)]  # x^i_k中标签为j的个数 / x^i_k的个数
                H_D_j = sum([-p * np.log2(p) if p != 0 else 0 for p in H_D_j])  # -∑(p_i * log2(p_i))
                H_D_Ai += v / len(input) * H_D_j  # ∑(|x^i_j| / |D| * H(D_j))
            g_D_A.append(H_D - H_D_Ai)   # H(D) - H(D|A_i)
        max_gain = g_D_A.index(max(g_D_A))
        node.feature_index = max_gain
        counter = Counter(input[:, max_gain])
        for k in counter.keys():
            node.next[k] = self.ID3(input[input[:, max_gain] == k], label[input[:, max_gain] == k],
                                    already_index + [max_gain])
        return node

    # 递归创建树
    # 不加连续值处理的
    def C4_5(self, input, label, already_index=[]):
        '''
        使用信息增益比来选择特征，使用西瓜书上的方法，先取出信息增益高于平均水平的特征，再从中选取信息增益比最大的特征作为下一次分类特征
        D代表整个数据集，特征分别为A_1、A_2、、、A_n，假设第i个特征的所有互异取值为x^i_1、x^i_2、x^i_3、、、x^i_n
        H(D) = -∑(p_i * log2(p_i))   这里的p_i代表第i个标签的统计概率,统计概率就是 i的个数/总个数

        信息增益：g(D, A_i) = H(D) - H(D|A_i) = H(D) - ∑(|x^i_j| / |D| * H(D_j))
        信息增益比：gain_ratio(D, A_i) = g(D, A_i) / H_A_i(D)
                 其中 H_A_i(D) = -∑(p_j * log2(p_j))   这里的p_i代表特征A_i的第j个取值的统计概率

        i代表第i个特征，j代表第i个特征的第j种取值，共有n种取值，|x^i_j|代表第i个特征的第j个取值个数，|D|代表数据集D的个数

        H(D_j) = -∑(p_i * log2(p_i))   这里的p_i = x^i_j中标签为k的个数 / x^i_j的个数  k=(0,1,2...)
        :param input: 输入训练数据，不会出现空的情况
        :param label:
        :param already_index: 记录已经进行过决策的特征
        :return: node
        '''
        node = TreeNode()
        cur_n_target = len(set(label))
        # 将当前节点的标签值设为所含样本最多的类别，不管子节点还是父节点，防止出现训练数据之外的特征值
        label_num = [sum(label == i) for i in range(self.n_target)]
        node.label = label_num.index(max(label_num))
        # 如果当前数据集全部是一样的标签
        if cur_n_target == 1:
            node.label = label[0]
            return node
        # 如果遍历过所有特征了
        if len(already_index) == self.feature_num:
            return node

        # 计算 H(D) = -∑(p_i * log2(p_i))
        H_D = [sum(label == c) / len(label) for c in set(label)]  # p_i
        H_D = sum([-p * np.log2(p) for p in H_D])  # -∑(p_i * log2(p_i))
        # 用来存储各个特征的信息增益
        g_D_A = []
        # 信息增益比
        g_D_A_ratio = []
        for i in range(input.shape[-1]):
            # 已经遍历过的特征无需再次计算
            if i in already_index:
                g_D_A.append(0)
                continue
            # 统计当前特征A_i各个取值的个数
            counter = Counter(input[:, i])
            # ∑(|x^i_j| / |D| * H(D_j)), 具体i,j,k字母有变化
            H_D_Ai = 0
            for k, v in counter.items():
                H_D_j = [sum(label[input[:, i] == k] == j) / v for j in range(cur_n_target)]  # x^i_k中标签为j的个数 / x^i_k的个数
                H_D_j = sum([-p * np.log2(p) if p != 0 else 0 for p in H_D_j])  # -∑(p_i * log2(p_i))
                H_D_Ai += v / len(input) * H_D_j  # ∑(|x^i_j| / |D| * H(D_j))
            g_D_A.append(H_D - H_D_Ai)   # H(D) - H(D|A_i)
            # H_A_i(D) = -∑(p_j * log2(p_j))   这里的p_i代表特征A_i的第j个取值的统计概率
            H_A_i = [v / len(input) for v in counter.values()]
            H_A_i = sum([-p * np.log2(p) for p in H_A_i])
            g_D_A_ratio.append(g_D_A[-1] / H_A_i)
        # 先取出信息增益高于平均水平的特征，将其设置为负无穷
        g_D_A = np.array(g_D_A)
        candidate_gain = [0 if i else -np.inf for i in g_D_A > g_D_A.mean()]
        # 然后加到信息增益比上，取最大值
        max_gain = np.argmax(np.array(g_D_A_ratio) + np.array(candidate_gain))
        node.feature_index = max_gain
        counter = Counter(input[:, max_gain])
        for k in counter.keys():
            node.next[k] = self.C4_5(input[input[:, max_gain] == k], label[input[:, max_gain] == k],
                                    already_index + [max_gain])
        return node

    # 递归创建树
    def C4_5_continuous(self, input, label, already_index=[], continuous=[]):
        '''
        使用信息增益比来选择特征，使用西瓜书上的方法，先取出信息增益高于平均水平的特征，再从中选取信息增益比最大的特征作为下一次分类特征
        D代表整个数据集，特征分别为A_1、A_2、、、A_n，假设第i个特征的所有互异取值为x^i_1、x^i_2、x^i_3、、、x^i_n
        H(D) = -∑(p_i * log2(p_i))   这里的p_i代表第i个标签的统计概率,统计概率就是 i的个数/总个数

        信息增益：g(D, A_i) = H(D) - H(D|A_i) = H(D) - ∑(|x^i_j| / |D| * H(D_j))
        信息增益比：gain_ratio(D, A_i) = g(D, A_i) / H_A_i(D)
                 其中 H_A_i(D) = -∑(p_j * log2(p_j))   这里的p_i代表特征A_i的第j个取值的统计概率

        对于连续值，先选取当前特征所有值，排好序，取出所有二分点作为候选分割点，按小于等于该值和大于该值将数据集分割成正集和负集，分别计算每个分割点的信息增益比
        选取最大的作为当前特征信息增益比，然后与其他特征比较

        i代表第i个特征，j代表第i个特征的第j种取值，共有n种取值，|x^i_j|代表第i个特征的第j个取值个数，|D|代表数据集D的个数

        H(D_j) = -∑(p_i * log2(p_i))   这里的p_i = x^i_j中标签为k的个数 / x^i_j的个数  k=(0,1,2...)
        :param input: 输入训练数据，不会出现空的情况
        :param label:
        :param already_index: 记录已经进行过决策的特征
        :return: node
        '''
        node = TreeNode()
        cur_n_target = list(set(label))

        # 将当前节点的标签值设为所含样本最多的类别，不管子节点还是父节点，防止出现训练数据之外的特征值
        label_num = [sum(label == i) for i in range(self.n_target)]
        node.label = label_num.index(max(label_num))
        # 如果当前数据集全部是一样的标签
        if len(cur_n_target) == 1:
            node.label = label[0]
            return node
        # 如果遍历过所有特征了
        if len(already_index) == self.feature_num:
            return node

        # 计算 H(D) = -∑(p_i * log2(p_i))
        H_D = [sum(label == c) / len(label) for c in set(label)]  # p_i
        H_D = sum([-p * np.log2(p) for p in H_D])  # -∑(p_i * log2(p_i))
        # 用来存储各个特征的信息增益
        g_D_A = []
        # 信息增益比
        g_D_A_ratio = []
        # 用来记录连续值变量的信息，包括分割值、分割数据集索引
        continuous_info = {}
        for i in range(input.shape[-1]):
            # 已经遍历过的特征无需再次计算
            if i in already_index:
                g_D_A.append(0)
                g_D_A_ratio.append(0)
                continue
            # 离散值
            if i not in continuous:
                # 统计当前特征A_i各个取值的个数
                counter = Counter(input[:, i])
                # ∑(|x^i_j| / |D| * H(D_j)), 具体i,j,k字母有变化
                H_D_Ai = 0
                for k, v in counter.items():
                    H_D_j = [sum(label[input[:, i] == k] == j) / v for j in cur_n_target]  # x^i_k中标签为j的个数 / x^i_k的个数
                    H_D_j = sum([-p * np.log2(p) if p != 0 else 0 for p in H_D_j])  # -∑(p_i * log2(p_i))
                    H_D_Ai += v / len(input) * H_D_j  # ∑(|x^i_j| / |D| * H(D_j))
                g_D_A.append(H_D - H_D_Ai)   # H(D) - H(D|A_i)
                # H_A_i(D) = -∑(p_j * log2(p_j))   这里的p_i代表特征A_i的第j个取值的统计概率
                H_A_i = [v / len(input) for v in counter.values()]
                H_A_i = sum([-p * np.log2(p) for p in H_A_i])
                g_D_A_ratio.append(g_D_A[-1] / H_A_i)
            # 连续值
            else:
                # 选取当前特征所有可取值，取出所有二分中值作为分割候选值
                features = sorted(list(set(input[:, i])))
                if len(features) == 1:
                    g_D_A.append(0)
                    g_D_A_ratio.append(0)
                    continue

                # 获取可取值的所有二分点作为候选分割点
                candidate_values = [(features[feature_i-1] + features[feature_i]) / 2 for feature_i in range(1, len(features))]
                split_candidate_gain = []
                split_candidate_gain_ratio = []
                # 对于每一个候选值计算增益
                for candidate in candidate_values:
                    # 先计算每个候选值分割后的正负集数量
                    input_split = [input[input[:, i] <= candidate], input[input[:, i] > candidate]]
                    label_split = [label[input[:, i] <= candidate], label[input[:, i] > candidate]]
                    # 正负子集的个数除以数据集总个数
                    collection_proportion = [len(input_split[0]) / len(input), len(input_split[1]) / len(input)]
                    # 每个正负集中各个标签的所占的比例
                    # 负集：[标签0比例，标签1比例，标签2比例]；正集：[标签0比例，标签1比例，标签2比例]；
                    H_D_j = [[sum(label_split[p_n] == j) / len(label_split[p_n]) for p_n in range(2)] for j in range(max(cur_n_target)+1)]
                    # [负集H-：-标签0比例*log2(标签0比例) - 标签1比例*log2(标签1比例) - 标签2比例*log2(标签2比例),
                    #  正集H+：-标签0比例*log2(标签0比例) - 标签1比例*log2(标签1比例) - 标签2比例*log2(标签2比例)]
                    a = [sum([-H_D_j[j][p_n] * np.log2(H_D_j[j][p_n]) if H_D_j[j][p_n] != 0 else 0 for j in cur_n_target]) for p_n in range(2)]
                    # 负集所占比例*负集H- + 正集所占比例*正集H+，得到该分割点的信息增益
                    a = sum([collection_proportion[p_n] * a[p_n] for p_n in range(2)])
                    # 计算信息增益比：先得到 -负集所占比例*log2(负集所占比例) - 正集所占比例*log2(正集所占比例)
                    H_A_i = sum([-p * np.log2(p) for p in collection_proportion])
                    split_candidate_gain.append(H_D - a)
                    split_candidate_gain_ratio.append((H_D - a) / H_A_i)
                # 取信息增益比最高的分割点作为该特征分割点
                split_index = np.argmax(np.array(split_candidate_gain_ratio))
                continuous_info[i] = {}
                continuous_info[i]['split_value'] = candidate_values[split_index]
                g_D_A.append(split_candidate_gain[split_index])
                g_D_A_ratio.append(split_candidate_gain_ratio[split_index])
        if len(g_D_A) != len(g_D_A_ratio):
            print()
        # 先取出信息增益高于平均水平的特征，将其设置为负无穷
        g_D_A = np.array(g_D_A)
        candidate_gain = [0 if i else -np.inf for i in g_D_A > g_D_A.mean()]
        # 然后加到信息增益比上，取最大值
        max_gain = np.argmax(np.array(g_D_A_ratio) + np.array(candidate_gain))
        node.feature_index = max_gain
        if max_gain in continuous_info:
            node.feature_continuous = True
            node.split_value = continuous_info[max_gain]['split_value']
            node.next['小于'] = self.C4_5_continuous(input[input[:, max_gain] <= node.split_value], label[input[:, max_gain] <= node.split_value],
                                    already_index, continuous=continuous)
            node.next['大于'] = self.C4_5_continuous(input[input[:, max_gain] > node.split_value], label[input[:, max_gain] > node.split_value],
                                    already_index, continuous=continuous)
        else:
            counter = Counter(input[:, max_gain])
            for k in counter.keys():
                node.next[k] = self.C4_5_continuous(input[input[:, max_gain] == k], label[input[:, max_gain] == k],
                                        already_index + [max_gain], continuous=continuous)
        return node

    def post_pruning(self, input_valid, label_valid, node=None):
        '''
        后剪枝操作，需要额外使用一个验证集进行剪枝，递归从叶子节点进行
        :param input_valid:
        :param label_valid:
        :param node: 初始时node为根节点
        :return:
        '''
        if not node:
            node = self.root
        # 如果是叶子节点，返回当前预测正确的数量
        if not node.next:
            accuracy_count = sum(np.array(label_valid) == node.label)
            return accuracy_count
        # 非叶子节点，计算当前预测正确的数量，与所有叶子节点预测正确的数量进行对比
        cur_accuracy_count = sum(np.array(label_valid) == node.label)
        child_accuracy_count = 0
        # 连续值
        if node.feature_continuous:
            for k, v in node.next.items():
                if k == '小于':
                    index = input_valid[:, node.feature_index] <= node.split_value
                else:
                    index = input_valid[:, node.feature_index] > node.split_value
                child_accuracy_count += self.post_pruning(input_valid[index], label_valid[index], v)
        # 离散值
        else:
            for k, v in node.next.items():
                index = input_valid[:, node.feature_index] == k
                child_accuracy_count += self.post_pruning(input_valid[index], label_valid[index], v)
        # 如果展开叶子节点判断正确的数量没有比不展开大，就进行剪枝
        if child_accuracy_count <= cur_accuracy_count:
            node.next = {}
            node.feature_index = -1
        return max(child_accuracy_count, cur_accuracy_count)

    def classify(self, input_test, label_test):
        input_test = self.to_numpy(input_test)
        label_test = self.to_numpy(label_test)
        pred = []
        for input_i in range(len(input_test)):
            node = self.root
            while True:
                if not node.next:
                    pred.append(node.label)
                    break

                # 连续值
                if node.feature_continuous:
                    if input_test[input_i][node.feature_index] <= node.split_value:
                        node = node.next['小于']
                    else:
                        node = node.next['大于']
                else:
                    # 离散值：对于未出现过训练集中的新特征值，直接使用父节点的标签，不再向下递归
                    if input_test[input_i][node.feature_index] in node.next:
                        node = node.next[input_test[input_i][node.feature_index]]
                    else:
                        pred.append(node.label)
                        break
        accuracy = self.accuracy(pred, label_test)
        return accuracy

    def to_numpy(self, x):
        if isinstance(x, list):
            x = np.array(x)
        return x

    def accuracy(self, pred, label):
        pred = self.to_numpy(pred)
        label = self.to_numpy(label)

        if len(pred.shape) != 1:
            pred = np.argmax(pred, axis=-1)
        return sum(pred == label) / pred.shape[0]


if __name__ == '__main__':
    iris = load_iris()

    X = iris.data
    y = iris.target
    print(X.data.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=57)
    # X_train = [['青年', '否', '否', '一般'],
    #            ['青年', '否', '否', '好'],
    #            ['青年', '是', '否', '好'],
    #            ['青年', '是', '是', '一般'],
    #            ['青年', '否', '否', '一般'],
    #            ['中年', '否', '否', '一般'],
    #            ['中年', '否', '否', '好'],
    #            ['中年', '是', '是', '好'],
    #            ['中年', '否', '是', '非常好'],
    #            ['中年', '否', '是', '非常好'],
    #            ['老年', '否', '是', '非常好'],
    #            ['老年', '否', '是', '好'],
    #            ['老年', '是', '否', '好'],
    #            ['老年', '是', '否', '非常好'],
    #            ['老年', '否', '否', '一般'],
    #            ]
    # y_train = [0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
    # scores = 0
    # for i in range(100):
    #     tree = Decision_Tree()
    #     tree.fit(X_train, y_train, 'ID3')
    #     scores += tree.classify(X_test, y_test)
    # print(scores/100)

    # scores = 0
    # for i in range(100):
    #     tree = Decision_Tree()
    #     tree.fit(X_train, y_train, 'C4.5')
    #     scores += tree.classify(X_test, y_test)
    # print(scores/100)

    tree = Decision_Tree()
    tree.fit(X_train, y_train, 'C4.5', continuous=[0,1,2,3])
    scores = tree.classify(X_test, y_test)
    print(scores)
    tree.post_pruning(X_test, y_test)
    scores = tree.classify(X_test, y_test)
    print(scores)

    tree = Decision_Tree()
    tree.fit(X_train, y_train, 'C4.5', continuous=[])
    scores = tree.classify(X_test, y_test)
    print(scores)
    tree.post_pruning(X_test, y_test)
    scores = tree.classify(X_test, y_test)
    print(scores)

