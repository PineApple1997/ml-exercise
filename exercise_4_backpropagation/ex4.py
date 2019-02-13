#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/2/13 21:58
# @Author  : qin yuxin
# @File    : ex4.py
# @Software: PyCharm

import numpy as np
import ex3
import ex3_nn
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1+np.exp(-z))


def deserialize(s, split, new_r, new_c, start):
    """
    反序列化 把序列s中的一段 变成new_r*new_c的矩阵
    """
    if start:
        return s[:split].reshape(new_r, new_c)
    else:
        return s[split:].reshape(new_r, new_c)


def serialize(a, b):
    """
    将a,b都变成一维向量并且合并
    """
    return np.concatenate((np.ravel(a), np.ravel(b)))


def forward_propagation(theta, X):
    theta1 = deserialize(theta, 25*401, 25, 401, start=True)
    theta2 = deserialize(theta, 25*401, 10, 26, start=False)
    a1 = X
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((a2.shape[0], 1)), a2), axis=1)
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)  # final h
    return a3


def cost():
    print("it's time to sleep")


def main():
    # raw_X, raw_y = ex3.load_data("ex4data1.mat")
    # ex3.plot_100_img(raw_X)
    raw_X, raw_y = ex3.load_data("ex4data1.mat", transpose=False)
    print("raw_y:", raw_y)
    y = ex3.expand_y(raw_y).T  # y's shape: 5000*10  0~9
    X = np.concatenate((np.ones((raw_X.shape[0], 1)), raw_X), axis=1)
    print("X's shape:", X.shape)
    print("y's shape:", y.shape)
    theta1, theta2 = ex3_nn.load_weight("ex4weights.mat")
    theta = serialize(theta1, theta2)
    print("theta's shape", theta.shape)

    h = forward_propagation(theta, X)
    print("h's shape:", h.shape)
    np.set_printoptions(suppress=True)
    print(h)



if __name__ == '__main__':
    main()