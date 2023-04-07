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
    # print('MSELoss:')
    # print('\ty: {}'.format(y.shape))
    # print('\ty_hat: {}'.format(y_hat.shape))
    error = y - y_hat
    # print('\terror: {}'.format(error.shape))
    loss = np.mean(error**2)
    # print('\tloss: {}'.format(loss.shape))
    activ_deriv = Activation(activ).f_deriv
    delta = -error * activ_deriv(y_hat)
    # print('\tdelta: {}'.format(delta.shape))
    return loss, delta

def softmax(x):
    exp = np.exp(x)
    return exp/np.sum(exp)

def CrossEntropyLoss(y, y_hat):
    # print(y_hat)
    # y_hat_sf = softmax(y_hat)
    # y_hat = np.clip(y_hat, 1e-12, 1. - 1e-12)
    # a = np.array([-1,2,3])
    # print(a)
    # print(softmax(a))
    # print(np.sum(softmax(a)))
    # print(y)
    # print(y_hat)
    # print(np.sum(y_hat))
    # print(y_hat[range(y.shape[0]),y.argmax(axis=1)])
    # print(-np.log(y_hat[range(y.shape[0]),y.argmax(axis=1)]))
    # print(np.sum(-np.log(y_hat[range(y.shape[0]),y.argmax(axis=1)])))
    # print(np.sum(-np.log(y_hat[range(y.shape[0]),y.argmax(axis=1)]))/y.shape[0])
    # log_liklihood = -np.log()
    loss = np.sum(-np.log(y_hat[range(y.shape[0]),y.argmax(axis=1)]))/y.shape[0]
    # print(loss)
    # print(y_hat)
    # print(np.sum(y_hat))
    # y_hat[range(y.shape[0]),y.argmax(axis=1)] -= 1
    # print(y_hat)
    # print(np.sum(y_hat))
    # delta = y_hat/y.shape[0]
    # delta = y_hat
    # print(y)
    delta = y_hat - y
    # print(delta)
    return loss, delta

def SoftmaxCrossEntropyLoss(y, y_hat):
    # print('SoftmaxCrossEntropyLoss:')
    # print('\ty: {}'.format(y.shape))
    # print('\ty_hat: {}'.format(y_hat.shape))
    y_sf = softmax(y_hat)
    # print('\tp: {}'.format(y_sf.shape))
    # print('\tnp.log(y_sf): {}'.format(np.log(y_sf).shape))
    loss = -np.sum(y * np.log(y_sf))/y.shape[0]
    # print('\tloss: {}'.format(loss))
    delta = y_sf - y
    # print('\tdelta: {}'.format(delta.shape))
    return loss, delta