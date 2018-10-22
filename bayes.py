#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np

from .field import *


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
        cs = []
        for k in range(0, len(df)):
            x = df.loc[df.index[k], [f.name for f in self.features]]
            cs.append(self.predict(x))
        return cs

    # def errordf(self, df, test, weights=None):
    #     if weights is None:
    #         pred = self.predictdf(df)
    #         e = np.mean([p - t for p, t in zip(pred, test)])
    #         L = len(test)
    #         new_weights = [1 / r / L if p==t else  r / L for p, t in zip(pred, test)]
    #     else:
    #         pred = self.predictdf(df)
    #         e = np.sum([(p - t) * weight for p, t, weight in zip(pred, test, weights)])
    #         new_weights = [w / r if p==t else w * r for p, t, w in zip(pred, test, weights)]
    #     r = np.sqrt((1 - e) / e)
    #     self.alpha = np.log(r)  # coef (weight) of classifier
    #     return e, new_weights / np.sum(new_weights)


    @classmethod
    def bleach(cls, x_train, y_train):
        raw_x = x_train
        raw_y = y_train
        x_train = []
        y_train = []
        for x, y in zip(raw_x, raw_y):
            if all(str(xi) != 'nan' for xi in x):
                x_train.append(x)
                y_train.append(y)
        return x_train, y_train


class ZeroOneClassifier(Classifier):
    def __init__(self, features, labelDist):
        super(ZeroOneClassifier, self).__init__([0, 1], features, labelDist)

    def errordf(self, df, test, weights=None):
        if weights is None:
            pred = self.predictdf(df)
            L = len(test)
            e = len([1 for predi, testi in zip(pred, test) if predi != testi]) / L
            r = np.sqrt((1 - e) / e)
            new_weights = [1 / r / L if p==t else  r / L for p, t in zip(pred, test)]
        else:
            pred = self.predictdf(df)
            e = np.sum([weight for predi, testi, weight in zip(pred, test, weights) if predi != testi])
            r = np.sqrt((1 - e) / e)
            new_weights = [w / r if p==t else w * r for p, t, w in zip(pred, test, weights)]
        self.alpha = np.log(r)  # coef (weight) of classifier
        z = 2 * e * r
        return e, new_weights / z


class BayesClassifier(Classifier):
    def __init__(self, labels, features, labelDist):
        super(BayesClassifier, self).__init__(labels, features, labelDist)
        self.eps = 1

    def totalProb(self, x):
        pass
    
    def postProb(self, c, x):
        return jointProb(self, x, c) / self.totalProb(x)

    def jointProb(self, x, c):
        return self.condProb(x, c) * self.labelDist[c]

    def log_jointProb(self, x, c):
        return self.log_condProb(x, c) - np.log(self.labelDist[c])

    def condProb(self, x, c):
        pass

    def predict(self, x):
        k = np.argmin([self.log_jointProb(x, c) for c in self.labels])
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

        nbc = cls(features, labelDist)
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
        p = self.log_jointProb(x, 0)
        q = self.log_jointProb(x, 1)
        return p > q

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

    @classmethod   
    def fromDataFrame(cls, x_train, y_train):
        pos_train = x_train[y_train==1]
        neg_train = x_train[y_train==0]
        return cls.fromPN(pos_train, neg_train)

class NaiveBayesClassifier(BayesClassifier):

    def condProb(self, x, c):
        # indepenence
        return np.prod([self._condProb(f, xi, c) for f, xi in zip(self.features, x)])

    def log_condProb(self, x, c):
        # indepenence
        return np.sum([self._log_condProb(f, xi, c) for f, xi in zip(self.features, x)])

    def _condProb(self, f, x, c):
        return self._jointProb(f, x, c) / self.labelDist[c]

    def _log_condProb(self, f, x, c):
        return -np.log(self._jointProb(f, x, c)) + np.log(self.labelDist[c])


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

    def jointProb_ratio(self, f, x, c1, c2):
        # p(x, c1) / p(x, c2), x is the feature f
        eps = self.eps
        xs = self.x_train[f.name]
        
        if f.is_discrete:
            n1 = n2 = 0
            for xi, ci in zip(xs, self.y_train):
                if xi == x:
                    if ci == c1:
                        n1 += 1
                    elif ci == c2:
                        n2 += 1
            p = (n1 + eps) / (n2 + eps)       # Bayes estimate
            return p
        else:
            n1 = n2 = 0
            for xi, ci in zip(xs, self.y_train):
                if str(x) == 'nan' and str(xi) == 'nan' or str(xi) != 'nan' and f.approx(xi, x):
                    if ci == c1:
                        n1 += 1
                    elif ci == c2:
                        n2 += 1
            p = (n1 + eps) / (n2 + eps)
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

    def jointProb_ratio(self, f, x):
        # p(x, 1) / p(x, 0) ~ N1(x) / N0(x)
        eps = self.eps
        
        if f.is_discrete:
            n1 = n2 = 0
            for xi in self.pos_train[f.name]:
                if xi == x:
                    n1 += 1
            for xi in self.neg_train[f.name]:
                if xi == x:
                    n2 += 1
            return (n1 + eps) / (n2 + eps)       # Bayes estimate
        else:
            n1 = n2 = 0
            for xi in self.pos_train[f.name]:
                if str(x) == 'nan':
                    if str(xi) == 'nan':
                        n1 += 1
                elif str(xi) != 'nan' and f.approx(xi, x):
                    n1 += 1
            for xi in self.neg_train[f.name]:
                if str(x) == 'nan':
                    if str(xi) == 'nan':
                        n2 += 1
                elif str(xi) != 'nan' and f.approx(xi, x):
                    n2 += 1
            return (n1 + eps) / (n2 + eps)


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


class StupidBayesClassifier(BayesClassifier):
    def __init__(self, labels, features, labelDist):
        super(StupidBayesClassifier, self).__init__(labels, features, labelDist)
        p0 = 0
        for l, p in labelDist.items():
            if p > p0:
                p0 = p
                self.best_class = l

    def predict(self, x):
        return self.best_class

class ZeroOneStupidBayesClassifier(ZeroOneBayesClassifier):
    def __init__(self, features, labelDist):
        super(ZeroOneStupidBayesClassifier, self).__init__(features, labelDist)
        if labelDist[0] > labelDist[1]:
            self.best_class = 0
        else:
            self.best_class = 1

    def predict(self, x):
        return self.best_class