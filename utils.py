__author__ = 'arenduchintala'
from math import exp, log, pi
import copy


def normpdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * pi * var) ** .5
    num = exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def flatten_backpointers(bt):
    reverse_bt = []
    while len(bt) > 0:
        x = bt.pop()
        reverse_bt.append(x)
        if len(bt) > 0:
            bt = bt.pop()
    reverse_bt.reverse()
    return reverse_bt


def logadd(x, y):
    """
    trick to add probabilities in logspace
    without underflow
    """
    if x == 0.0 and y == 0.0:
        return log(exp(x) + exp(y))  # log(2)
    elif x >= y:
        return x + log(1 + exp(y - x))
    else:
        return y + log(1 + exp(x - y))


def logadd_of_list(a_list):
    logsum = a_list[0]
    for i in a_list[1:]:
        logsum = logadd(logsum, i)
    return logsum


def gradient_checking(theta, eps, val):
    f_approx = {}
    for i in theta:
        theta_plus = copy.deepcopy(theta)
        theta_minus = copy.deepcopy(theta)
        theta_plus[i] = theta[i] + eps
        theta_minus[i] = theta[i] - eps
        f_approx[i] = (val(theta_plus) - val(theta_minus)) / (2 * eps)
    return f_approx
