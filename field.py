#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from collections import Iterable, Counter

import numpy as np
import numpy.linalg as LA

def _min(xs):
    return min(x for x in xs if x!=0)

def _max(xs):
    return max(x for x in xs if x!=0)

class Field(object):
    '''Field has 3 (principal) propteries
    name: name
    type_: type 
    range_: range  [None]'''
    defaultPart = 20

    def __init__(self, name='none', type_=str, range_=None):
        self.name = name
        self.type_ = type_
        self.range_ = range_
        self.is_hybrid = False
        if type_ == float:
            self.is_discrete = False
            if self.range_:
                self.part = Field.defaultPart
            else:
                self.step = 1
        else:
            self.is_discrete = True
        self.dim = 1

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
        '''Infer the type from a value
        
        Arguments:
            value {anything} -- any value
        
        Keyword Arguments:
            name {str} -- the name of the field (default: {'none'})
        
        Returns:
            Field
        '''

        if isinstance(value, str):
            return Field(name, str)
        elif isinstance(value, (int, np.int64)):
            return Field(name, int)
        elif isinstance(value, (float, np.float64)):
            return Field(name, float)
        elif isinstance(value, Iterable):
            t = Field.get_type(value)
            f = Field(name, t)
            f.dim = len(value)
            f.is_hybrid = True
            f.types = set(t)
            return f
        else:
            return Field(name, type(value))

    @staticmethod
    def get_type(value):
        if isinstance(value, str):
            return str
        elif isinstance(value, (int, np.int64)):
            return int
        elif isinstance(value, (float, np.float64)):
            return float
        elif isinstance(value, Iterable):
            return tuple(map(Field.get_type, value))

    @staticmethod
    def fromValues(values, name='none', continuous=False):
        if values:
            f = Field.fromValue(values[0], name)
            if f.is_continuous or continuous:
                if isinstance(values[0], Iterable):
                    A = np.array([value for value in values if 0 not in value])
                    f.range_ = A.min(axis=0), A.max(axis=0)
                    # stds = np.array([np.std([v[k] for v in values]) for k in range(f.dim)])
                    f.part = Field.defaultPart
                    f.is_hybrid = True
                else:
                    f.range_ = _min(values), _max(values)
                    f.part = Field.defaultPart
                f.is_discrete = False
            return f
        else:
            return Field(name, float)

    @staticmethod
    def fromValuesx(values, name='none', tol=0.01):

        if isinstance(values[0], int):  # regarded as a continous variable
            c = Counter(values)
            if np.mean([n for a, n in c.items()]) / len(values) < tol:
                return Field.fromValues(values, name, continuous=True)
            else:
                return Field.fromValues(values, name)
        elif isinstance(values[0], str):
            return Field.fromValues(values, name)
        elif isinstance(values[0], Iterable):
            return Field.fromValues(values, name, continuous=True)
        else:
            return Field.fromValues(values, name)

    def __str__(self):
        return self.name

    def approx(self, a, b, step=None):
        if step is None:
            step = self.step
        if self.is_discrete:
            return a == b
        elif self.is_hybrid:
            return np.all(abs(ai - bi) < s for ai, bi, s in zip(a, b, self.step))
        else:
            return abs(a - b) < self.step
