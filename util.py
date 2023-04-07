from base import *
from math import log

class Activation(object):
    def __init__(self, activ='tanh'):
        if activ == 'tanh':
            self.f = self._tanh
            self.f_deriv = self._tanh_deriv
        elif activ == 'logistic':
            self.f = self._logistic
            self.f_deriv = self._logistic_deriv
        elif activ == 'relu':
            self.f = self._relu
            self.f_deriv = self._relu_deriv
        elif activ == 'softmax':
            self.f = self._softmax
            self.f_deriv = None
        else:
            self.f = None
            self.f_deriv = None

    def _tanh(self, x):
        x = np.tanh(x)
        return x

    def _tanh_deriv(self, x):
        return 1.0 - x**2

    def _logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _logistic_deriv(self, x):
        return  x * (1 - x)

    def _relu(self, x):
        return x * (x > 0)

    def _relu_deriv(self, x):
        return 1 * (x > 0)

    def _softmax(self, x):
        exp = np.exp(x)
        return exp/np.sum(exp)

def MSELoss(y, y_hat, activ='tanh'):
    error = y - y_hat
    loss = np.mean(error**2)
    activ_deriv = Activation(activ).f_deriv
    delta = -error * activ_deriv(y_hat)
    return loss, delta

def softmax(x):
    exp = np.exp(x)
    return exp/np.sum(exp)

def CrossEntropyLoss(y, y_hat):
    loss = np.sum(-np.log(y_hat[range(y.shape[0]),y.argmax(axis=1)]))/y.shape[0]
    delta = y_hat - y
    return loss, delta

def SoftmaxCrossEntropyLoss(y, y_hat):
    y_sf = softmax(y_hat)
    loss = -np.sum(y * np.log(y_sf))/y.shape[0]
    delta = y_sf - y
    return loss, delta