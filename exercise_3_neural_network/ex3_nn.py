#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/2/13 13:30
# @Author  : qin yuxin
# @File    : ex3_nn.py
# @Software: PyCharm


import numpy as np
import scipy.io as scio
from sklearn.metrics import classification_report
import ex3
import ex2


def load_weight(path):
    data = scio.loadmat(path)
    theta1 = data['Theta1']
    theta2 = data['Theta2']
    return theta1, theta2


def main():
    theta1, theta2 = load_weight('ex3weights.mat')  # (25, 401) (10, 26)
    print(theta1.shape, theta2.shape)

    raw_X, raw_y = ex3.load_data('ex3data1.mat', transpose=False)
    X = np.concatenate((np.ones((raw_X.shape[0], 1)), raw_X), axis=1)

    a1 = X  # input unit on layer 1  a1.shape: 5000*401
    z2 = a1 @ theta1.T  # z2.shape: 5000*25
    a2 = ex2.sigmoid(z2)  # a2.shape: 5000*25
    a2 = np.concatenate((np.ones((a2.shape[0], 1)), a2), axis=1)  # add a2[0]=1, a2.shape:5000*26
    z3 = a2 @ theta2.T  # z3.shape: 5000*10
    a3 = ex2.sigmoid(z3)  # a3.shape : 5000*10
    h = a3

    y_pred = np.argmax(h, axis=1) + 1
    accuracy = np.mean(y_pred == raw_y)
    print('accuracy = {0}%'.format(accuracy * 100))
    print(classification_report(raw_y, y_pred))


if __name__ == '__main__':
    main()