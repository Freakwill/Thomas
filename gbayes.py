#!/usr/local/bin/python
# -*- coding: utf-8 -*-

'''Generalized Bayes Classifiers
'''
import copy

import numpy as np
from sklearn import svm
from sklearn.linear_model import LassoLars

from .field import *
from .bayes import *

from neupy import algorithms, environment

environment.reproducible()

model_dict={'svm':svm.SVC(kernel='rbf', gamma='scale'), 'grnn':algorithms.GRNN(verbose=False), 'pnn':algorithms.PNN(verbose=False)}
model_class_dict={'svm':svm.SVC, 'grnn':algorithms.GRNN, 'pnn':algorithms.PNN, 'lasso':LassoLars}

class SemiNaiveBayesClassifier(BayesClassifier):
    pass

class ZeroOneSemiNaiveBayesClassifier(ZeroOneNaiveBayesClassifier, SemiNaiveBayesClassifier):
    '''0-1 SemiNaiveBayesClassifier
    
    Principle:
    P(c|X,Y) ~ P(X|c) P(c|Y) ~ prod_i p(x_i | c) fc(Y)
    
    Extends:
        ZeroOneNaiveBayesClassifier
        SemiNaiveBayesClassifier
    '''

    def condProb(self, x, z, c):
        # P(c|X,Y) ~ P(X|c) P(c|Y) ~ prod_i p(x_i | c) fc(Y)
        p1 = np.prod([self._condProb(f, xi, c) for f, xi in zip(self.features, x)])
        p2 = self._predict(z, c)
        return p1 * p2

    def _predict(self, z, c):
        p = self.model.predict([z])[0][0]
        return p if c else 1-p

    def predict(self, x, z):
        p = self.log_condProb(x, z, 0)
        q = self.log_condProb(x, z, 1)
        return p > q

    @classmethod   
    def fromPN(cls, pos_train1, neg_train1, z_train, y_train):
        '''
        a neural network will be trained with z_train, y_train
        
        Arguments:
            pos_train1, neg_train1 same in super class
            z_train {DateFrame}
            y_train {Array}
        
        Returns:
            ZeroOneSemiNaiveBayesClassifier
        '''
        
        sbc = super(ZeroOneSemiNaiveBayesClassifier, cls).fromPN(pos_train1, neg_train1)
        nn = algorithms.GRNN(std=np.std(z_train.values), verbose=False)
        nn.train(z_train, y_train)
        sbc.model = nn
        sbc.features2 = z_train.columns

        return sbc

    @classmethod   
    def fromDataFrame(cls, x_train, z_train, y_train):
        # Call fromPN
        pos_train = x_train[y_train==1]
        neg_train = x_train[y_train==0]
        return cls.fromPN(pos_train, neg_train, z_train, y_train)

    def predictdf(self, df):
        cs = []
        for k in range(0, len(df)):
            x = df.loc[df.index[k], [f.name for f in self.features]]
            z = df.loc[df.index[k], self.features2]
            cs.append(self.predict(x, z))
        return cs


class ZeroOneHemiNaiveBayesClassifier(ZeroOneNaiveBayesClassifier, SemiNaiveBayesClassifier):
    '''0-1 HemiNaiveBayesClassifier
    
    Principle:
    P(c|X,Y1, Y2) ~ P(X|c) P(c|Y) ~ prod_i p(x_i | c) fc(Y1)gc(Y2)
    
    Extends:
        ZeroOneNaiveBayesClassifier
        SemiNaiveBayesClassifier
    '''

    def condProb(self, x, zs, c):
        # P(c|X,Y) ~ P(X|c) P(c|Y) ~ prod_i p(x_i | c) fc(Y)
        p1 = np.prod([self._condProb(f, xi, c) for f, xi in zip(self.features, x)])
        p2 = self._predict(zs, c)
        return p1 * p2

    def log_condProb(self, x, zs, c):
        # P(c|X,Y) ~ P(X|c) P(c|Y) ~ prod_i p(x_i | c) fc(Y)
        l1 = np.sum([self._log_condProb(f, xi, c) for f, xi in zip(self.features, x)])
        l2 = - np.log(self._predict(zs, c))
        return l1 + l2


    def ratio2(self, zs):
        ps = np.array([model.predict([z]) for model, z in zip(self.models, zs) if np.all(list(map(lambda x: str(x)!='nan' and not np.isnan(x), z)))]).ravel()
        ps = np.array([p for p in ps if not np.isnan(p)])
        return np.prod(ps / (1 - ps))

    def predict(self, x, zs):
        # prod_i N1(xi) / N0(xi) * (f(y)/(1-f(y)) * (N0 / N1) ** n
        p = self.ratio2(zs)
        n = len(self.features)
        m = len(self.features2)
        r = np.prod([self.jointProb_ratio(f, xi) for xi, f in zip(x, self.features)]) * p * (self.labelDist[0] / self.labelDist[1]) ** (n + m -1)
        return r > 1

    # def predict(self, x, zs):
    #     p = self.log_condProb(x, zs, 0)
    #     q = self.log_condProb(x, zs, 1)
    #     return p > q

    @classmethod   
    def fromPN(cls, pos_train, neg_train, z_trains, y_train, models=None):
        '''
        a neural network will be trained with z_trains, y_train
        
        Arguments:
            pos_train, neg_train same in super class
            z_trains {List[DateFrame]}
            y_train {Array}
        
        Returns:
            ZeroOneSemiNaiveBayesClassifier
        '''
        
        sbc = super(ZeroOneHemiNaiveBayesClassifier, cls).fromPN(pos_train, neg_train)
        if models is None or models == 'grnn':
            sbc.models = [algorithms.GRNN(std=np.std([a for a in z_train.values.ravel() if str(a)!='nan' and a!=0]), verbose=False) for z_train in z_trains]
        elif models == 'pnn':
            sbc.models = [algorithms.PNN(std=np.std([a for a in z_train.values.ravel() if str(a)!='nan' and a!=0]), verbose=False) for z_train in z_trains]
        elif models == 'svm':
            sbc.models = [svm.SVC(kernel='rbf') for z_train in z_trains]
        elif models == 'lasso':
            sbc.models = [LassoLars() for z_train in z_trains]
        else:
            sbc.models = [copy.deepcopy(model_dict[model]) if isinstance(model, str) else copy.deepcopy(model) for model in models]
        sbc.features2 = [z_train.columns for z_train in z_trains]
        sbc.fit(z_trains, y_train)
        return sbc

    @classmethod   
    def fromDataFrame(cls, x_train, z_trains, y_train, models=None):
        # Call fromPN
        pos_train = x_train[y_train==1]
        neg_train = x_train[y_train==0]
        return cls.fromPN(pos_train, neg_train, z_trains, y_train, models=models)

    def predictdf(self, df):
        cs = []
        for k in range(0, len(df)):
            x = df.loc[df.index[k], [f.name for f in self.features]]
            zs = [df.loc[df.index[k], fs] for fs in self.features2]
            cs.append(self.predict(x, zs))
        return cs

    def fit(self, z_trains, y_train):
        for model, z_train in zip(self.models, z_trains):
            z_train, y_train = ZeroOneHemiNaiveBayesClassifier.bleach(z_train.values, y_train)
            if hasattr(model, 'fit'):
                model.fit(z_train, y_train)
            else:
                model.train(z_train, y_train)

