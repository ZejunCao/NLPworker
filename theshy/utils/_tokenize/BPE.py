#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/3/25 1:11
# @File    : BPE.py
# @Software: PyCharm
# @System  : Windows
# @desc    : null

import json
import re
from collections import defaultdict
from theshy.utils._tokenize.common import BasicTokenizer

# 代码参考: https://leimao.github.io/blog/Byte-Pair-Encoding/#Byte-Pair-Encoding-Algorithm
# 代码参考：https://github.com/rsennrich/subword-nmt
# paper: https://arxiv.org/abs/1508.07909


class BPE:
    def __init__(self, vocab=None):
        self.vocab = vocab
        self.basic_tokenizer = BasicTokenizer()
        self.unknown_token = '[UNK]'

    def build_vocab(self, corpus, max_iter=100):
        vocab = self.get_word_freq(corpus)
        for i in range(max_iter):
            pairs = self.get_stats(vocab)
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            if pairs[best] == 1:
                break
            vocab = self.merge_vocab(best, vocab)

        vocab = self.get_token(vocab)
        vocab = list(vocab.keys())
        vocab.remove('</w>')
        self.vocab = {k: i for i, k in enumerate(vocab)}

    def tokenize(self, string):
        text_list = self.basic_tokenizer.tokenize(string)
        sorted_tokens = list(self.vocab)
        sorted_tokens.sort(key=lambda x: -self.measure_token_length(x))

        res = []
        for text in text_list:
            res.extend(self.tokenize_word(text + '</w>', sorted_tokens))

        return res

    def tokenize_word(self, word, sorted_tokens):
        if not word:
            return []
        if sorted_tokens == []:
            return [self.unknown_token]

        res = []
        for i in range(len(sorted_tokens)):
            token = sorted_tokens[i]
            token_re = re.escape(token)

            pos = [(m.start(0), m.end(0)) for m in re.finditer(token_re, word)]
            if len(pos) == 0:
                continue
            all_end_pos = [p[0] for p in pos]

            single_start_pos = 0
            for single_end_pos in all_end_pos:
                subword = word[single_start_pos: single_end_pos]
                res += self.tokenize_word(word=subword, sorted_tokens=sorted_tokens[i + 1:])
                res += [token]
                single_start_pos = single_end_pos + len(token)
            remaining_subword = word[single_start_pos:]
            res += self.tokenize_word(word=remaining_subword, sorted_tokens=sorted_tokens[i + 1:])
            break
        else:
            return [self.unknown_token]
        return res

    # 统计语料库中每个单词出现的频率
    def get_word_freq(self, corpus):
        '''
        :param corpus: List = [sentence1, sentence2, ...]
        :return: Dict  = {word1: freq1, word2, freq2, ...}, word中每个字母用空格分开，且结尾加上 </w>
        '''
        word_vocab = defaultdict(int)
        for text in corpus:
            text = self.basic_tokenizer.tokenize(text)
            for t in text:
                word_vocab[' '.join(list(t)) + ' </w>'] += 1

        return word_vocab

    def get_token(self, word_vocab):
        token_vocab = defaultdict(int)
        for word, freq in word_vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                token_vocab[token] += freq

        return token_vocab

    # 得到每一字母对的统计
    def get_stats(self, vocab):
        '''
        :param word_vocab: Dict  = {word1: freq1, word2, freq2, ...}
        :return: Dict  = {(letter1, letter2): freq1, (letter2, letter3): freq2, ...}
        把相邻的两个字母组成对，统计同时出现的频率。letter：字母
        '''
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbol = word.split()
            for i in range(len(symbol)-1):
                pairs[symbol[i], symbol[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        """
        EXAMPLE:
            word = 'T h e <\w>'
            pair = ('e', '<\w>')
            word_after_merge = 'T h e<\w>'

        输入:
            pair: Tuple[str, str] # 需要合并的字符对
            v_in: Dict[str, int]  # 合并前的vocab
        输出:
            v_out: Dict[str, int] # 合并后的vocab
        注意:
            当合并word 'Th e<\w>'中的字符对 ('h', 'e')时，'Th'和'e<\w>'字符对不能被合并。
        """
        v_out = {}
        # 把pair拆开，然后用空格合并起来，然后用\把空格和其他符号转义
        bigram = re.escape(' '.join(pair))
        # 自定义一个正则规则, (?<!\S)h\ e(?!\S) 只有前面、后面不是非空白字符(\S)(意思前后得是没东西的)，才匹配h\ e，这样就可以把Th\ e<\w>排除在外
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]

        return v_out


    def measure_token_length(self, token):
        if token[-4:] == '</w>':
            return len(token[:-4]) + 1
        else:
            return len(token)

if __name__ == '__main__':
    corpus = ['Tokenization is often used to protect credit card data',
              'What is Tokenization? Tokenization is a process by which PANs',
              'Tokenization is the process of transforming ',
              ' The token is a randomized data string missing. form formulate reform record refine return',]
    bpe_tokenizer = BPE()
    bpe_tokenizer.build_vocab(corpus, max_iter=100)
    a = bpe_tokenizer.tokenize(corpus[0])
    print(bpe_tokenizer.tokenize(corpus[0]))
    # vocab = {}
    # for k, v in bpe_tokenizer.vocab.items():
    #     if k[-4:] == '</w>':
    #         vocab[k[:-4]] = v
    #     else:
    #         vocab[k] = v

    # v_path = '/data/zhangxiao585/pycharm_data/layoutlmv3-base-fintuned-funsd/vocab.json'
    # with open(v_path, 'r', encoding='utf-8') as f:
    #     v = json.load(f)
    # tokenizer = FullTokenizer(v)

    text = "Figure 3: Diagram今天天气真好what's are you talking about?"
    # text = "dasdasform missing.  asd dasda what's are you ??"
    # print(tokenizer.tokenize(text))
    print(bpe_tokenizer.tokenize(text))

    print()
