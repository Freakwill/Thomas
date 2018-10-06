#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

# To get some scores, it requires mystat
from mystat import *

train = pd.read_excel('test.xlsx') # DataFrame

# data for train
x_train = train.iloc[:7, 1:]
y_train = train.iloc[:7, 0]

# data for test
x_test = train.iloc[7:, 1:]
y_test = train.iloc[7:, 0]

import thomas

def nb():
    
    # so easy
    nbc = thomas.NaiveBayesClassifier.fromDataFrame(x_train, y_train)
    
    # predict
    y_pred = [nbc.predict(x) for x in x_test.values]

    # calculate scores
    scores = check(y_test, y_pred)
    print(scores)
    print('f-score(p)', fscore(*scores))
    print('f-socre(n)', fnscore(*scores))
    print('mcc', mcc(*scores))


nb()