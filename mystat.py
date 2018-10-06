#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np

def hmean(p, r):
    return 2 * p * r / (p + r)


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
    # print(tp, fp, fn, tn)
    return tp, fp, fn, tn

def fscore(tp, fp, fn, tn):
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    return hmean(P, R)

def fnscore(tp, fp, fn, tn):
    P = tn / (tn + fn)
    R = tn / (tn + fp)
    return hmean(P, R)


def mcc(tp, fp, fn, tn):
    # Matthews correlation coefficient
    return (tp * tn - fp * fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fn)*(tn+fp))


def joint(x, y):
    c0 = []
    c1 = []
    cs = []
    for xi, yi in zip(x, y):
        if yi == 0:
            if xi in cs:
                k = cs.index(xi)
                c0[k] += 1
            else:
                c0.append(1)
                c1.append(0)
                cs.append(xi)
        else:
            if xi in cs:
                k = cs.index(xi)
                c1[k] += 1
            else:
                c1.append(1)
                c0.append(0)
                cs.append(xi)
    return c0, c1

def jointx(x, y, cs=[], step=1):
    def in_(x, cs):
        for k, c in enumerate(cs):
            if abs(x-c) < step:
                return k
        return -1

    c0 = np.zeros(len(cs))
    c1 = np.zeros(len(cs))
    for xi, yi in zip(x, y):
        if yi == 0:
            k = in_(xi, cs)
            if  k != -1:
                c0[k] += 1
            else:
                c0 = np.hstack((c0, [1]))
                c1 = np.hstack((c1, [0]))
                cs.append(xi)
        else:
            k = in_(xi, cs)
            if k != -1:
                c1[k] += 1
            else:
                c1 = np.hstack((c1, [1]))
                c0 = np.hstack((c0, [0]))
                cs.append(xi)
    return c0, c1
