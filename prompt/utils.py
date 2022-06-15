#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/30 21:22
# software: PyCharm-utils

from common import load_jsonl, save_list


def load_data(data_file):
    """将原始输入转换成prompt text"""
    items = load_jsonl(data_file)
    data = []
    for item in items:
        subj = item['subject']
        obj = item['object']
        contact = item['predicate'].split('_')[1]  # subject和object的关系
        if 'salience' in item:
            row = '{}{}{}吗 {}'.format(subj, contact, obj, item['salience'])
        else:
            row = '{}{}{}吗'.format(subj, contact, obj)
        data.append(row)
    return data


if __name__ == '__main__':
    train_data_all = load_data('../data/train_triple.jsonl')
    test = load_data('../data/dev_triple.jsonl')

    offset = int(len(train_data_all) * 0.1)
    train, dev = train_data_all[offset:], train_data_all[:offset]

    save_list(train, '../data/prompt/train.txt')
    save_list(dev, '../data/prompt/dev.txt')
    save_list(test, '../data/prompt/test.txt')
