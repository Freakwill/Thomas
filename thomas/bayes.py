#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from thomas.field import *


class Classifier(object):
    '''Classifier has 2 (principal) propteries
    labels: labels
    features: features'''
    def __init__(self, labels, features, labelDist={}):
        self.labels = labels
        self.features = features
        self.labelDist = labelDist
        self.jointProbDict = {}

    def predictdf(self, df):
        return [self.predict(row) for k, row in df.iterrows()]


class ZeroOneClassifier(Classifier):
    def __init__(self, features, labelDist):
        super(ZeroOneClassifier, self).__init__([0, 1], features, labelDist)


class BayesClassifier(Classifier):
    def __init__(self, labels, features, labelDist):
        super(BayesClassifier, self).__init__(labels, features, labelDist)
        self.eps = 0.55

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
        
        features = [Field.fromValuesx([_ for _ in x_train[key] if str(_) != 'nan'], key) for key in x_train.columns]

        labels = list(set(y_train))

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

class ZeroOneBayesClassifier(ZeroOneClassifier, BayesClassifier):

    def predict_with_prob(self, x):
        p = self.jointProb(x, 0)
        q = self.jointProb(x, 1)
        if p > q:
            return 0, p / self.labelDist[0]
        else:
            return 1, q / self.labelDist[1]

    def predict(self, x):
        p = self.jointProb(x, 0)
        q = self.jointProb(x, 1)
        return p < q

    @classmethod   
    def fromPN(cls, pos_train, neg_train):
        
        features = []
        for key in pos_train.columns:
            f = Field.fromValuesx([_ for _ in pos_train[key] if str(_) != 'nan'], key)
            features.append(f)
        for key in neg_train.columns:
            if key not in pos_train.columns:
                f = Field.fromValuesx([_ for _ in neg_train[key] if str(_) != 'nan'], key)
                features.append(f)

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
        eps = self.eps
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
                    elif str(xi) != 'nan' and f.approx(xi, x):
                        n += 1
            p = (n + eps) / (N + (f.part + 1) ** f.dim * eps)
            return p


    def __setstate__(self, state):
        self.lables, self.features, self.labelDist, self.jointProbDict = state

    def __getstate__(self, state):
        return self.lables, self.features, self.labelDist, self.jointProbDict


class ZeroOneNaiveBayesClassifier(ZeroOneBayesClassifier, NaiveBayesClassifier):

    def _jointProb(self, f, x, c):
        eps = self.eps
        N = len(self.pos_train) + len(self.neg_train)
        if c == 1:
            xs = self.pos_train[f.name]
        else:
            xs = self.neg_train[f.name]
        
        if f.is_discrete:
            if f.name in self.jointProbDict and (x, c) in self.jointProbDict[f.name]:
                return self.jointProbDict[f.name][(x, c)]
            n = 0
            for xi in xs:
                if xi == x:
                    n += 1
            k = len(set(xs))
            p = (n + eps) / (N + k * eps)       # Bayes estimate
            if f.name not in self.jointProbDict:
                self.jointProbDict[f.name] = {}
            self.jointProbDict[f.name].setdefault((x, c), p)
            return p
        else:
            n = 0
            for xi in xs:
                if str(x) == 'nan':
                    if str(xi) == 'nan':
                        n += 1
                elif str(xi) != 'nan' and f.approx(xi, x):
                    n += 1
            p = (n + eps) / (N + (f.part + 1) ** f.dim * eps)
            return p

    def plot(self, feature1, feature2, axes=None):
        from matplotlib.font_manager import FontProperties
        myfont = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
        p = self.pos_train[[feature1, feature2]]
        n = self.neg_train[[feature1, feature2]]
        import matplotlib.pyplot as plt
        if axes is None:
            axes = plt.figure().add_subplot(111)
        axes.plot(p, '+b', label='pos')
        axes.plot(n, '.r', label='neg')
        axes.set_xlabel(feature1, fontproperties=myfont)
        axes.set_ylabel(feature2, fontproperties=myfont)
        plt.show()

    def plot3D(self, feature1, feature2, feature3, axes=None):
        from matplotlib.font_manager import FontProperties
        myfont = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
        px, py, pz = self.pos_train[feature1], self.pos_train[feature2], self.pos_train[feature3]
        nx, ny, nz = self.neg_train[feature1], self.neg_train[feature2], self.neg_train[feature3]
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        if axes is None:
            axes = Axes3D(plt.figure())
        axes.plot3D(px, py, pz, '+b', label='pos')
        axes.plot3D(nx, ny, nz, '.r', label='neg')
        axes.set_xlabel(feature1, fontproperties=myfont)
        axes.set_ylabel(feature2, fontproperties=myfont)
        axes.set_zlabel(feature3, fontproperties=myfont)
        plt.show()
