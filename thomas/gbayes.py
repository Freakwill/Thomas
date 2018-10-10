#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from thomas.field import *
from thomas.bayes import *

from neupy import algorithms, environment

environment.reproducible()

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
        p = self.condProb(x, z, 0)
        q = self.condProb(x, z, 1)
        return p < q

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
        grnn = algorithms.GRNN(std=0.5, verbose=False)
        grnn.train(z_train, y_train)
        sbc.model = grnn
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
