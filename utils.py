import numpy as np
import math

class Epoch_Params_Grad_Holder:
    # list wrapper that holds last 2 (epoch, params, grad) tuples for calculating lipz coeffs
    def __init__(self):
        self.__data = []
        self.__old = None
    
    def add(self, new):
        if type(new) is not tuple or len(new) != 3:
            raise RuntimeError('can only add 3-tuples')
        if len(self.__data) < 2:
            self.__data.append(new)
        else:
            self.__old = self.__data[0]
            self.__data[0] = self.__data[1]
            self.__data[1] = new

    def len(self):
        return len(self.__data) 

    def prev_epoch(self):
        if len(self.__data) == 0:
            raise RuntimeError('object empty')
        if len(self.__data) == 1:
            return('no previous epoch')
        return self.__data[0][0]

    def prev_params(self):
        if len(self.__data) == 0:
            raise RuntimeError('object empty')
        if len(self.__data) == 1:
            raise RuntimeError('no previous params')
        return self.__data[0][1]

    def prev_grad(self):
        if len(self.__data) == 0:
            raise RuntimeError('object empty')
        if len(self.__data) == 1:
            raise RuntimeError('no previous gadient')
        return self.__data[0][2]


    def epoch(self):
        if len(self.__data) == 0:
            raise RuntimeError('object empty')
        if len(self.__data) == 1:
            return self.__data[0][0]
        return self.__data[1][0]

    def params(self):
        if len(self.__data) == 0:
            raise RuntimeError('object empty')
        if len(self.__data) == 1:
            return self.__data[0][1]
        return self.__data[1][1]

    def grad(self):
        if len(self.__data) == 0:
            raise RuntimeError('object empty')
        if len(self.__data) == 1:
            return self.__data[0][2]
        return self.__data[1][2]

    def lipz_coeff(self):
        if len(self.__data) < 2:
            return 0
        if np.array_equal(self.params(), self.prev_params()):
            return math.inf
        return np.linalg.norm(self.grad() - self.prev_grad())/np.linalg.norm(self.params() - self.prev_params())

    def revert(self):
        self.__data[1] = self.__data[0]
        self.__data[0] = self.__old



