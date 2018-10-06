#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from collections import Iterator
from toolz import unique

def check(y_true, y_pred):
    tp = fp = fn = tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1:
            if yp == 1:
                tp += 1
            else:
                fp += 1
        else:
            if yp == 1:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn


class Field(object):
    '''Field has 3 (principal) propteries
    name: name
    type_: type 
    range_: range  [None]'''
    defaultPart = 10
    def __init__(self, name='none', type_=str, range_=None):
        self.name = name
        self.type_ = type_
        self.range_ = range_
        if type_ == float:
            self.is_discrete = False
            if self.range_:
                self.part = Field.defaultPart
            else:
                self.step = 1
        else:
            self.is_discrete = True

    @property
    def is_continuous(self):
        return not self.is_discrete
    

    @property
    def part(self):
        return self._part

    @part.setter
    def part(self, s):
        self._part = s
        self.step = (self.range_[1] - self.range_[0]) / s
    

    @staticmethod
    def fromValue(value, name='none'):
        if isinstance(value, str):
            return Field(name, str)
        elif isinstance(value, (int, np.int64)):
            return Field(name, int, (0, 100))
        elif isinstance(value, (float, np.float64)):
            return Field(name, float, (0, 100))
        elif isinstance(value, Iterator):
            f = Field(name, Iterator)
            f.dim = len(value)
            f.hybrid = True
            return f
        else:
            return Field(name, type(value))

    @staticmethod
    def fromValues(values, name='none', continuous=False):
        if values:
            f = Field.fromValue(values[0], name)
            if not f.is_discrete:
                f.range_ = min(values), max(values)
            elif continuous:
                f.range_ = min(values), max(values)
                f.is_discrete = False
                f.part = Field.defaultPart
            return f
        else:
            return Field(name, float)

    @staticmethod
    def fromValuesx(values, name='none', tol=0.01):

        if isinstance(values[0], int) and max(values) / len(values) < tol:  # regarded as a continous variable
            return Field.fromValues(values, name, continuous=True)
        else:
            return Field.fromValues(values, name)

    def __str__(self):
        return self.name


class Classifier(object):
    '''Classifier has 2 (principal) propteries
    labels: labels
    features: features'''
    def __init__(self, labels, features, labelDist={}):
        self.labels = labels
        self.features = features
        self.labelDist = labelDist
        self.jointProbDict = {}


class OneZeroClassifier(Classifier):
    def __init__(self, features, labelDist):
        super(OneZeroClassifier, self).__init__([0, 1], features, labelDist)


class BayesClassifier(Classifier):
    def __init__(self, labels, features, labelDist):
        super(BayesClassifier, self).__init__(labels, features, labelDist)

    def totalProb(self, x):
        pass
    
    def postProb(self, c, x):
        return jointProb(self, x, c) / self.totalProb(x)

    def jointProb(self, x, c):
        return self.condProb(x, c) * self.labelDist[c]

    def condProb(self, x, c):
        pass

    def predict(self, x):
        k = np.argmax([self.jointProb(x, c) for c in self.labels])
        cx = self.labels[k]
        return cx

    def predict_with_prob(self, x):
        p = [self.jointProb(x, c) for c in self.labels]
        k = np.argmax(p)
        cx = self.labels[k]

        return cx, p[k] / self.labelDist[c]

    @classmethod   
    def fromDataFrame(cls, x_train, y_train):
        
        features = [Field.fromValuesx([_ for _ in x_train.loc[:, key] if str(_) != 'nan'], key) for key in x_train.columns]

        labels = list(unique(y_train))

        N = len(y_train)
        labelDist = {}
        for l in labels:
            labelDist.setdefault(l, 0)
            for y in y_train:
                if y == l:
                    labelDist[l] += 1
            labelDist[l] /= N

        nbc = cls(labels, features, labelDist)
        nbc.x_train = x_train
        nbc.y_train = y_train
        return nbc

class OneZeroBayesClassifier(BayesClassifier, OneZeroClassifier):

    def predict_with_prob(self, x):
        p = self.jointProb(x, 0)
        q = self.jointProb(x, 1)
        if q > q:
            return 0, p / self.labelDist[0]
        else:
            return 1, q / self.labelDist[1]

    @classmethod   
    def fromDataFrame(cls, pos_train, neg_train):
        
        features = [Field.fromValues([_ for _ in x_train.loc[:, key] if str(_) != 'nan'], key) for key in unique(pos_train.columns + neg_train.columns)]

        Np = len(pos_train)
        Nn = len(neg_train)
        N = Np + Nn
        labelDist = [Nn / N, Np / N]

        nbc = cls(features, labelDist)
        nbc.pos_train = pos_train
        nbc.neg_train = neg_train
        return nbc

class NaiveBayesClassifier(BayesClassifier):

    def condProb(self, x, c):
        # indepenence
        return np.prod([self._condProb(f, xi, c) for f, xi in zip(self.features, x)])


    def _condProb(self, f, x, c):
        return self._jointProb(f, x, c) / self.labelDist[c]


    def _jointProb(self, f, x, c):
        eps = 0.00005
        N = len(self.y_train)
        xs = self.x_train[f.name]
        
        if f.is_discrete:
            if f.name in self.jointProbDict and (x, c) in self.jointProbDict[f.name]:
                return self.jointProbDict[f.name][(x, c)]
            n = 0
            for xi, ci in zip(xs, self.y_train):
                if ci == c and xi == x:
                    n += 1
            k = len(set(xs))
            p = (n + eps) / (N + k * eps)       # Bayes estimate
            if f.name not in self.jointProbDict:
                self.jointProbDict[f.name] = {}
            self.jointProbDict[f.name].setdefault((x, c), p)
            return p
        else:
            n = 0
            for xi, ci in zip(xs, self.y_train):
                if ci == c:
                    if str(x) == 'nan':
                        if str(xi) == 'nan':
                            n += 1
                    elif str(xi) != 'nan' and abs(xi - x) < f.step:
                        n += 1
            p = (n + eps) / (N + f.part * eps)
            return p


    def test(self, x_test, y_test):
        y_pred = [self.predict(x) for x in x_test]
        return check(y_test, y_pred)

    def __setstate__(self, state):
        self.lables, self.features, self.labelDist, self.jointProbDict = state

    def __getstate__(self, state):
        return self.lables, self.features, self.labelDist, self.jointProbDict



class OneZeroNaiveBayesClassifier(OneZeroBayesClassifier, NaiveBayesClassifier):
    pass
