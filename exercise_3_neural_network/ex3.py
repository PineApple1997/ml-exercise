#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/2/12 15:54
# @Author  : qin yuxin
# @File    : ex3.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import ex2_reg
import ex2
from sklearn.metrics import classification_report


def load_data(path, transpose=True):
    data = scio.loadmat(path)
    raw_X = data['X']  # 特征值
    if transpose:  # 翻转
        raw_X = np.array([im.reshape((20, 20)).T for im in raw_X])  # 把图像顺时间转90度
        raw_X = np.array([im.reshape(400) for im in raw_X])  # 把图像翻转
    raw_y = data['y']  # 结果（0～9）
    raw_y = raw_y.reshape(raw_y.shape[0])
    print("raw X's shape:", raw_X.shape)
    print("raw y's shape:", raw_y.shape)
    return raw_X, raw_y


def plot_one_img(X):
    """
    image : (400,1)
    """
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(X.reshape((20, 20)), cmap='gray_r')  # cmap是color map 这里选用灰度图像
    plt.xticks(np.array([]))  # 去除边界刻度
    plt.yticks(np.array([]))
    plt.show()


def plot_100_img(X):
    """
    image : (400,100)
    """
    chosen_lines = np.random.choice(np.arange(X.shape[0]), 100)
    chosen = X[chosen_lines, :]
    fig, ax = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(10, 10))

    for r in range(10):
        for c in range(10):
            ax[r, c].matshow(chosen[r * 10 + c].reshape((20, 20)), cmap='gray_r')
    plt.xticks(np.array([]))  # 去除边界刻度
    plt.yticks(np.array([]))
    plt.show()


def expand_y(raw_y):
    y_matrix = []
    for k in range(1, 11):  # 原数据中是y把0写作10 把它修正为0（matlab中起始为1）
        # 每次append的是一个list, y_matrix[i]有5000行，表示y中每个数是否等于i，等于为1，不等为0
        y_matrix.append((raw_y == k).astype(int))   # y_matrix: 10*5000
    y_matrix = [y_matrix[-1]] + y_matrix[:-1]  # [y_matrix[-1]]是为10的那些y 放到第一列（就变成0了
    y = np.array(y_matrix)
    return y


def main():
    """
    一共有5000个训练样本，每个样本是20*20像素的数字的灰度图像。
    每个像素代表一个浮点数，表示该位置的灰度强度。
    20×20的像素网格被展开成一个400维的向量。
    X，每一个样本都变成了一行，这给了我们一个5000×400矩阵X，每一行都是一个手写数字图像的训练样本。
    """
    raw_X, raw_y = load_data("ex3data1.mat")

    # 单个图像
    pick_one = np.random.randint(0, 5000)
    plot_one_img(raw_X[pick_one])
    print('this should be', raw_y[pick_one])

    # 随机取100个样本
    # plot_100_img(X)

    # 准备数据
    X = np.concatenate((np.ones((raw_X.shape[0], 1)), raw_X), axis=1)  # 为x加上最前面为1的一列
    print("X.shape after add a column:", X.shape)
    y = expand_y(raw_y)
    print("new y's shape", y.shape)  # should be 10*5000

    # 训练一维模型 以0为例
    print("y[0]", y[0])
    y_0 = y[0].reshape((y[0].shape[0], 1))
    weight = ex2_reg.logistic_regression(X, y_0)  # 默认lambda=1 y[0] = 5000个样本是否为0的bool值
    y_pred = ex2.predict(X, weight)
    print('Accuracy={}'.format(np.mean(y[0] == y_pred)))

    # 多元分类
    # k_theta = np.array([ex2_reg.logistic_regression(X, y[k]) for k in range(10)])  # 列表生成式 生成k维theta
    k_theta = np.zeros((10, X.shape[1]))  # k_theta's shape: 10 * 401
    for k in range(10):
        tmp_y_k = y[k].reshape((y[k].shape[0], 1))
        k_theta[k] = ex2_reg.logistic_regression(X, tmp_y_k)

    prob_matrix = ex2.sigmoid(X @ k_theta.T)  # 假设函数 5000 * 10
    # 返回假设函数每行（每个数字）的最大theta的索引（多分类就是这么做的）
    # 对每个h(x)都进行prediction 然后取max 这样y_pred就是最终的5000*1的
    y_pred = np.argmax(prob_matrix, axis=1)  # 返回沿轴axis最大值的索引，axis=1代表行
    y_answer = raw_y.copy()
    y_answer[y_answer == 10] = 0  # 把10的那部分写为0
    print(classification_report(y_answer, y_pred))


if __name__ == '__main__':
    main()
