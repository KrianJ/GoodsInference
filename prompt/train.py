#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: KrianJ
# datetime:2022/5/30 22:12
# software: PyCharm-train

import os
import warnings
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

warnings.filterwarnings("ignore")


def train(model, train_loader, dev_loader, output_dir, device='cpu'):
    """Preparation"""
    best_path = os.path.join(output_dir, 'best_model.pth')  # 最优模型
    checkpoint_dir = os.path.join(output_dir, 'checkpoint')  # 创建checkpoint目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device(device)
    model.to(device)
    epoches = 2
    learning_rate = 1e-5
    weight_decay = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    """Starting Training"""
    model.train()
    true_labels = []
    pred_labels = []
    train_losses = []
    best_f1 = 0.  # 记录最好的F1值
    for epoch in range(epoches):
        for i, batch in enumerate(train_loader):
            batch.to(device)
            label = batch.label  # 真实标签
            logits = model(batch)  # 预测值
            preds = torch.argmax(logits, dim=-1)
            loss = criterion(logits, label)  # 交叉熵

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            label, preds = label.cpu().numpy(), preds.cpu().numpy()
            loss = float(loss.detach().cpu().numpy())
            true_labels.extend(label)
            pred_labels.extend(preds)
            train_losses.append(loss)
            # 计算得分
            acc = accuracy_score(label, preds)
            precision = precision_score(label, preds)
            recall = recall_score(label, preds)
            f1 = f1_score(label, preds)

            if i % 10 == 0:
                print("Epoch{}-{}, Loss: {}, Acc: {}, P: {}, R: {}, F1: {}".
                      format(epoch, i, loss, acc, precision, recall, f1))
            if i % 100 == 0 and i:  # 建立checkpoint
                dev_acc, dev_p, dev_r, dev_f1 = evaluate(model, criterion, dev_loader, device)
                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_{}_{}.pth'.format(epoch, dev_f1))
                    torch.save(model, checkpoint_path)

        acc = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        print('Epoch {} Finished\nTraining: acc: {}, p: {}, r: {}, f1: {}'.format(epoch, acc, precision, recall, f1))

        # evaluate
        dev_acc, dev_p, dev_r, dev_f1 = evaluate(model, criterion, dev_loader, device)
        print("Evaluate: acc:{}, p: {}, r: {}, f1: {}".format(dev_acc, dev_p, dev_r, dev_f1))
        # 保存模型
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model, best_path)


def evaluate(model, criterion, dev_loader, device):
    model.to(device)
    model.eval()
    true_labels = []
    pred_labels = []
    dev_loss = []
    with torch.no_grad():
        for batch in dev_loader:
            batch.to(device)
            labels = batch.label
            logits = model(batch)
            preds = torch.argmax(logits, dim=-1)
            loss = float(criterion(logits, labels).cpu().numpy())

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            dev_loss.append(loss)
    acc = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    return acc, precision, recall, f1
