#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import pandas as pd

from mystat import *

from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split


train = pd.read_excel('modeling_data.xls', encoding='utf-8')

train['批次时间'] = [_.total_seconds() for _ in train['批次完成时间'] - train['批次开始时间']]
train['L-W'] = list(zip(train['长度'], train['重量'], train['批次时间']))
train['K-M'] = list(zip(train['克重'], train['门幅']))
train = train.drop(columns=['批次开始时间', '批次完成时间', '配方ID', '流程卡号', '缸号'])


# train.fillna(0)
y_train = train['质量问题']
# select train data and test data from samples randomly
x_train, x_test, y_train, y_test = train_test_split(train, y_train, test_size=0.2)

z_trains = x_train[[s for s in train.columns if s.startswith('助剂')]], x_train[[s for s in train.columns if s.startswith('燃料')]], x_train[[s for s in train.columns if s.startswith('光谱值')]]
x_train = x_train[['机器', '弹力', '氨纶', '织物', '纱线', '颜色', '客户', '月份', 'L-W', 'K-M']]


import thomas

def nb():
    models = []
    nbc = thomas.ZeroOneHemiNaiveBayesClassifier.fromDataFrame(x_train, z_trains, y_train, models=('pnn', 'pnn', 'pnn'))

    y_pred = nbc.predictdf(x_test)
    scores = check(y_test, y_pred)
    print(report(scores))

nb()
