# Thomas
My Bayes algorithm, for the name of Thomas Bayes

## Features
* Cope with continuous random varaibles intellegently.
* integer random varaibles (e.g. the mass of things with integer gram) will be treated as continuous ones in some case.

## Requirement
* numpy
* pandas
* scikit-learn (in examples)
* neupy

## Install

`pip install tomas`
not `thomas` which had been regested.

## Why
For the Honor of T. Bayes

![](https://github.com/Freakwill/Thomas/blob/master/Thomas_Bayes.gif)


## Grammar
Just see the example file.

## Examples

```python

import pandas as pd

from mystat import *

from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

train = pd.read_excel('modeling_data.xls', encoding='utf-8')

train['批次时间'] = [_.total_seconds() for _ in train['批次完成时间'] - train['批次开始时间']]
train['G-W-T'] = list(zip(train['克重'], train['重量'], train['批次时间']))
train['L-M'] = list(zip(train['长度'], train['门幅']))
train = train.drop(columns=['批次开始时间', '批次完成时间', '配方ID', '流程卡号', '缸号', '批次时间'])

y_train = train['质量问题']

x_train, x_test, y_train, y_test = train_test_split(train, y_train, test_size=0.2)

# seperate the data in x_train to 1 + 3 groups as x_train and z_trains

x_train = x_train[['机器', '弹力', '氨纶', '织物', '纱线', '颜色', '客户', '月份', 'L-M', 'G-W-T']]
z_trains = x_train[[s for s in train.columns if s.startswith('助')]], x_train[[s for s in train.columns if s.startswith('染')]], x_train[[s for s in train.columns if s.startswith('光')]]

import thomas

def nb():
    models = None # use PNN to fit data (z_trains, y_train)
    nbc = thomas.ZeroOneHemiNaiveBayesClassifier.fromDataFrame(x_train, z_trains, y_train, models)
    y_pred = nbc.predictdf(x_test)
    scores = check(y_test, y_pred)
    print(report(scores))

nb()

# =>
   | -> C1 | -> C0|
    C1 | 128 | 43 |
    C0 | 217 | 431 |
----------------
    f-score(p) 0.4961
    f-socre(n) 0.7683
    mcc 0.3405
```

## Is it easy?
Yes

## Principle

![](https://github.com/Freakwill/Thomas/blob/master/README.png)
