#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from mystat import *

train = pd.read_excel('test.xlsx')

x_train = train.iloc[:7, 1:]
y_train = train.iloc[:7, 0]

x_test = train.iloc[7:, 1:]
y_test = train.iloc[7:, 0]

import thomas

def nb():
    
    nbc = thomas.NaiveBayesClassifier.fromDataFrame(x_train, y_train)

    y_pred = [nbc.predict(x) for x in x_test.values]
    scores = check(y_test, y_pred)
    print(scores)
    print('f-score(p)', fscore(*scores))
    print('f-socre(n)', fnscore(*scores))
    print('mcc', mcc(*scores))


nb()