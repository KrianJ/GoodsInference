#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/27 22:02
# software: PyCharm-utils

# -*- coding: utf-8 -*-
import json
import re
import time
import jsonlines
from hanziconv import HanziConv
from datetime import timedelta


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def load_jsonl(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = [row for row in jsonlines.Reader(f)]
    return data


def sentence_process(query):
    """预处理文本序列"""

    query = HanziConv.toSimplified(query.strip())
    reg_ex = re.compile('[\\W]+')  # 我们可以使用正则表达式来切分句子，切分的规则是除单词，数字外的任意字符串
    res = re.compile(r'([\u4e00-\u9fa5])')  # [\u4e00-\u9fa5]中文范围
    sentences = reg_ex.split(query.lower())
    str_list = []
    for sentence in sentences:
        if not res.split(sentence):
            str_list.append(sentence)
        else:
            ret = res.split(sentence)
            str_list.extend(ret)
    return [w for w in str_list if len(w.strip()) > 0]


def save_list(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in obj:
            if isinstance(item, str):
                f.write(item + '\n')
            else:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
