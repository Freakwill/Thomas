#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import pandas as pd

from mystat import *

from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split


train = pd.read_excel('modeling_data.xls', encoding='utf-8')

train['L-W'] = [(l, w) for l, w in zip(train['长度'], train['重量'])]
train['批次时间'] = [_.total_seconds() for _ in train['批次完成时间'] - train['批次开始时间']]

# train.fillna(0)
y_train = train['质量问题']
x_train = x_train[['机器', '弹力', '氨纶', '织物', '纱线', '颜色', '月份', '克重', '门幅', 'L-W', '批次时间']]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)


import thomas

def nb():
    
    nbc = thomas.ZeroOneNaiveBayesClassifier.fromPN(pos_train, neg_train)

    y_pred = [nbc.predict(x) for x in x_test.values]
    scores = check(y_test, y_pred)
    print(scores)
    print('f-score(p)', fscore(*scores))
    print('f-socre(n)', fnscore(*scores))
    m = mcc(*scores)
    print('mcc', m)


nb()

