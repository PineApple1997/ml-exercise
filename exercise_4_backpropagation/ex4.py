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
import scipy.optimize as opt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def expand_y(y):
    """
    expand 5000*1 into 5000*10
    where y=10 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]: ndarray
    最后一行是0（原数据是10）
    """
    res = []
    for i in y:
        y_array = np.zeros(10)
        y_array[i - 1] = 1
        res.append(y_array)
    return np.array(res)


def serialize(a, b):
    """
    将a,b都变成一维向量并且合并
    """
    return np.concatenate((np.ravel(a), np.ravel(b)))


def deserialize(seq):
    """
    into ndarray of (25, 401), (10, 26)
    """
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)


def forward_propagation(theta, X):
    theta1, theta2 = deserialize(theta)  # 第一列是1
    a1 = X
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)  # 第二层的unit
    a2 = np.concatenate((np.ones((a2.shape[0], 1)), a2), axis=1)  # 第二层的unit加上一个bias unit
    z3 = a2 @ theta2.T
    a3 = sigmoid(z3)  # final h
    return a1, z2, a2, z3, a3


def cost(theta, X, y):
    # theta = theta.reshape((X.shape[1], 1))
    _, _, _, _, h = forward_propagation(theta, X)
    m = X.shape[0]
    return np.sum((-y * np.log(h) - (1 - y) * np.log(1 - h))) / m


def regularized_cost(theta, X, y, l=1):
    """
    注意：不计算theta1[0]和theta2[0]
    """
    m = X.shape[0]
    theta1, theta2 = deserialize(theta)  # 第一列是1
    reg_t1 = np.sum(theta1[:, 1:] ** 2)
    reg_t2 = np.sum(theta2[:, 1:] ** 2)
    cost_term = cost(theta, X, y)
    regularized_term = l / (2 * m) * (reg_t1 + reg_t2)
    return cost_term + regularized_term


def gradient_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def rand_init_weights(size):
    return np.random.uniform(-0.12, 0.12, size)


def gradient(theta, X, y):
    """
    :return: 不计算正则项的梯度值
    """
    theta1, theta2 = deserialize(theta)  # 第一列是1
    a1, z2, a2, z3, h = forward_propagation(theta, X)
    # a1: 5000*10, z2: 5000*25, a2: 5000*26, z3: 5000*10, h: 5000*10
    # X: 5000*401, y:5000*10, theta1: 25*401, theta2: 10*26
    delta3 = h - y  # 5000 * 10
    m = X.shape[0]  # m=5000

    delta2 = (delta3 @ theta2[:, 1:]) * gradient_sigmoid(z2)  # delta2: (5000*25) * (5000*25) = (5000*25)
    D2 = delta3.T @ a2  # (10, 26)
    D1 = delta2.T @ a1  # (25, 401) 保持和theta1 theta2一样的纬度
    D = serialize(D1, D2) / m  # (10285,)
    return D


def regularized_gradient(theta, X, y, l=1):
    """
    don't regularize theta of bias terms
    注意：不计算theta1[0]和theta2[0]的bias term（正则项
    :return: 计算正则项的梯度值
    """
    m = X.shape[0]
    delta1, delta2 = deserialize(gradient(theta, X, y))  # 先得到不算正则项的梯度值
    t1, t2 = deserialize(theta)

    t1[:, 0] = 0  # 让第一列=0（原来为1  就是第一列不设置正则项 让其为0
    t2[:, 0] = 0
    delta1 = delta1 + (l / m) * t1
    delta2 = delta2 + (l / m) * t2

    return serialize(delta1, delta2)


def gradient_check(theta, X, y, e):
    def a_numeric_grad(plus, minus):
        """
        对每个参数theta_i计算数值梯度，即理论梯度。
        """
        return (regularized_cost(plus, X, y) - regularized_cost(minus, X, y)) / (e * 2)

    numeric_grad = []
    for i in range(len(theta)):
        plus = theta.copy()  # deep copy otherwise you will change the raw theta
        minus = theta.copy()
        plus[i] = plus[i] + e
        minus[i] = minus[i] - e
        grad_i = a_numeric_grad(plus, minus)
        numeric_grad.append(grad_i)

    numeric_grad = np.array(numeric_grad)  # 写成矩阵
    analytic_grad = regularized_gradient(theta, X, y)  # 我们算出来的梯度项（导数
    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad + analytic_grad)  # 二范数

    print("If your back propagation implementation is correct,\n"
          "the relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\n"
          "Relative Difference: {}".format(diff))


def nn_training(X, y):
    init_theta = rand_init_weights(10285)  # 25*401 + 10*26
    # res = opt.fmin_tnc(func=regularized_cost,
    #                    x0=init_theta,
    #                    fprime=regularized_gradient,
    #                    args=(X, y, 1))
    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 400})
    final_theta = res.x
    return final_theta


def show_accuracy(theta, X, y):
    _, _, _, _, h = forward_propagation(theta, X)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y, y_pred))


def plot_hidden_layer(theta):
    """
    theta: (10285, )
    """
    final_theta1, _ = deserialize(theta)
    hidden_layer = final_theta1[:, 1:]  # ger rid of bias term theta

    fig, ax_array = plt.subplots(nrows=5, ncols=5, sharey=True, sharex=True, figsize=(5, 5))

    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(hidden_layer[5 * r + c].reshape((20, 20)),
                                   cmap='gray_r')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()


def main():
    # raw_X, raw_y = ex3.load_data("ex4data1.mat")
    # ex3.plot_100_img(raw_X)
    raw_X, raw_y = ex3.load_data("ex4data1.mat", transpose=False)
    X = np.concatenate((np.ones((raw_X.shape[0], 1)), raw_X), axis=1)
    print("raw_y:", raw_y)
    y = expand_y(raw_y)  # y's shape: 5000*10  0~9
    print("X's shape:", X.shape)
    print("y's shape:", y.shape)
    theta1, theta2 = ex3_nn.load_weight("ex4weights.mat")
    theta = serialize(theta1, theta2)
    print("theta's shape", theta.shape)

    _, _, _, _, h = forward_propagation(theta, X)
    print("h's shape:", h.shape)
    # np.set_printoptions(suppress=True)
    print(h)
    c = cost(theta, X, y)
    print("cost:", c)
    reg_cost = regularized_cost(theta, X, y)
    print("reg_cost:", reg_cost)
    # D = gradient("gradient without reg", theta, X, y)
    # gradient_check(theta, X, y, e=0.0001)
    final_theta = nn_training(X, y)
    print("final_theta: ", final_theta)
    show_accuracy(final_theta, X, raw_y)
    plot_hidden_layer(final_theta)


if __name__ == '__main__':
    main()
