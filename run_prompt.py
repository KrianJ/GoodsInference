#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/30 21:52
# software: PyCharm-prompt

from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
from prompt.train import train as prompt_train

classes = ['0', '1']

# 加载原始数据
train = [
    InputExample(guid=i, text_a=text.split(' ')[0], label=int(text.split(' ')[1]))
    for i, text in enumerate(open('data/prompt/train.txt', 'r', encoding='utf-8'))
]
dev = [
    InputExample(guid=i, text_a=text.split(' ')[0], label=int(text.split(' ')[1]))
    for i, text in enumerate(open('data/prompt/dev.txt', 'r', encoding='utf-8'))
]
test = [
    InputExample(guid=i, text_a=text.split(' ')[0])
    for i, text in enumerate(open('data/prompt/test.txt', 'r', encoding='utf-8'))
]
# 加载预训练模型
model_path = r'F:\py_source\tianchi_GoodsInference\pretrained\bert-base-chinese'
plm, tokenizer, model_config, WrapperClass = load_plm("bert", model_path)
# 定义模板
promptTemplate = ManualTemplate(
    text='{"placeholder":"text_a"}回答是{"mask"}',
    tokenizer=tokenizer,
)
# 定义verbalizer
promptVerbalizer = ManualVerbalizer(
    classes=classes,
    label_words={
        "0": ["对"],
        "1": ["错"],
    },
    tokenizer=tokenizer,
)
# 组合以上结构
promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer,
)
# 数据加载
train_loader = PromptDataLoader(
    dataset=train,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    max_seq_length=32,
    batch_size=32
)
dev_loader = PromptDataLoader(
    dataset=dev,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    max_seq_length=32,
    batch_size=32
)
test_loader = PromptDataLoader(
    dataset=test,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    max_seq_length=32,
    batch_size=32
)

if __name__ == '__main__':
    prompt_train(promptModel, train_loader, dev_loader, 'output/prompt')


