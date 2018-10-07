#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from collections import Iterator
from toolz import unique

from thomas.field import *

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


def report(scores):
    s = '''
    | -> C1 | -> C0|
    C1|%d|%d|
    C0 |%d | %d |''' % scores
    return s


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
        '''Create a classifier from DataFrame
        
        [description]
        
        Arguments:
            x_train {DataFrame} -- Features
            y_train {List or Array-Like object} -- Classes
        
        Returns:
            BayesClassifier
        '''
        
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
        eps = 0.005
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
            p = (n + eps) / (N + (f.part + 1) * eps)
            return p


    def __setstate__(self, state):
        self.lables, self.features, self.labelDist, self.jointProbDict = state

    def __getstate__(self, state):
        return self.lables, self.features, self.labelDist, self.jointProbDict



class OneZeroNaiveBayesClassifier(OneZeroBayesClassifier, NaiveBayesClassifier):
    pass
