#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : Cao Zejun
# @Time    : 2023/3/25 1:10
# @File    : basicTokenizer.py
# @Software: PyCharm
# @System  : Windows
# @desc    : 用空格或特殊符号将英文和中文分割成单词或字

import unicodedata


class BasicTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        '''

        :param text: str; 例如：'今天太阳真好'
        :return:
        '''
        res = []
        text = self._tokenize_chinese_chars(text) # 把中文左右加空格，方便切分
        text = self._run_split_on_punc(text) # 把标点符号左右加空格，方便切分
        text = text.lower().split()
        for char in text:
            if char in [' ', '\t', '\n', '\r']:
                res.append(' ')
            else:
                res.append(char)

        return res

    def _tokenize_chinese_chars(self, text):
        res = []
        for char in text:
            if self._is_chinese_char(char):
                res.append(" ")
                res.append(char)
                res.append(" ")
            else:
                res.append(char)
        return "".join(res)

    def _is_chinese_char(self, char):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        cp = ord(char)
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _run_split_on_punc(self, text):
        res = []
        for char in text:
            if unicodedata.category(char).startswith("P"):
                res.append(" ")
                res.append(char)
                res.append(" ")
            else:
                res.append(char)

        return "".join(res)
