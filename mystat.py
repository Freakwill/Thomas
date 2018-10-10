#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np

def hmean(p, r):
    return 2 * p * r / (p + r)


def check(y_test, y_pred):
    '''Calculate TP, FP, FN and TN
    
    Arguments:
        y_test {Array} -- y-value for test
        y_pred {Array} -- y-value predicted by x_test
    
    Returns:
        tuple -- TP, FP, FN, TN
    '''
    tp = fp = fn = tn = 0
    for yt, yp in zip(y_test, y_pred):
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


def report(scores):
    s = '''
    | -> C1 | -> C0|
    C1 | %d | %d |
    C0 | %d | %d |''' % scores

    s += '''\n----------------
    f-score(p) %.4f
    f-socre(n) %.4f
    mcc %.4f''' % (fscore(*scores), fnscore(*scores), mcc(*scores))
    return s

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
    def in_(x, cs, step):
        for k, c in enumerate(cs):
            if abs(x-c) < step:
                return k
        return -1

    c0 = np.zeros(len(cs))
    c1 = np.zeros(len(cs))
    for xi, yi in zip(x, y):
        if yi == 0:
            k = in_(xi, cs, step)
            if  k != -1:
                c0[k] += 1
            else:
                c0 = np.hstack((c0, [1]))
                c1 = np.hstack((c1, [0]))
                cs.append(xi)
        else:
            k = in_(xi, cs, step)
            if k != -1:
                c1[k] += 1
            else:
                c1 = np.hstack((c1, [1]))
                c0 = np.hstack((c0, [0]))
                cs.append(xi)
    return c0, c1

def jointc(x, y, cs1=[], cs2=[], step1=1, step2=1):
    def in_(x, cs, step):
        for k, c in enumerate(cs):
            if abs(x-c) < step:
                return k
        return -1

    L1 = len(cs1)
    L2 = len(cs2)
    cs = np.zeros((L1, L2))

    for xi, yi in zip(x, y):
        k1 = in_(xi, cs1, step1)
        k2 = in_(yi, cs2, step2)
        if  k1 != -1:
            if k2 != -1:
                cs[k1, k2] += 1
            else:
                a = np.zeros((L1, 1)); a[k1, 0] = 1
                cs = np.hstack((cs, a))
                cs2.append(yi)
        else:
            if k2 != -1:
                a = np.zeros((1, L2)); a[0, k2] = 1
                cs = np.hstack((cs, a))
                cs1.append(xi)
            else:
                a = np.zeros((1, L2))
                cs = np.hstack((cs, a))
                a = np.zeros((L1 + 1, 1)); a[L1, 0] = 1
                cs = np.hstack((cs, a))
                cs1.append(xi)
                cs2.append(yi)
    return cs
