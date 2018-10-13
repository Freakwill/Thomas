#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import pandas as pd

from mystat import *

from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split


train = pd.read_excel('modeling_data.xls', encoding='utf-8')

train['批次时间'] = [_.total_seconds() for _ in train['批次完成时间'] - train['批次开始时间']]
train['G-W-T'] = list(zip(train['克重'], train['重量'], train['批次时间']))
train['L-M'] = list(zip(train['长度'], train['门幅']))
train = train.drop(columns=['批次开始时间', '批次完成时间', '配方ID', '流程卡号', '缸号', '批次时间'])


# train.fillna(0)
y_train = train['质量问题']
# select train data and test data from samples randomly
x_train, x_test, y_train, y_test = train_test_split(train, y_train, test_size=0.2)

z_trains = x_train[[s for s in train.columns if s.startswith('助')]], x_train[[s for s in train.columns if s.startswith('染')]], x_train[[s for s in train.columns if s.startswith('光')]]
x_train = x_train[['机器', '弹力', '氨纶', '织物', '纱线', '颜色', '客户', '月份', 'L-M', 'G-W-T']]

# from keras.models import Sequential
# from keras.layers.core import Dense, Activation

# lm = Sequential([
#     Dense(30, input_shape=(3,)),
#     Activation('relu'),
#     Dense(1),
#     Activation('sigmoid')])
# lm.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')


import thomas

def nb():
    models = None
    nbc = thomas.ZeroOneHemiNaiveBayesClassifier.fromDataFrame(x_train, z_trains, y_train, models)
    y_pred = nbc.predictdf(x_test)
    scores = check(y_test, y_pred)
    print(report(scores))

nb()
