#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/2/12 15:54
# @Author  : qin yuxin
# @File    : ex3.py
# @Software: PyCharm

import numpy as np
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import scipy.io as scio
from exercise_2_longistic_regression import ex2_reg


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


'''
一共有5000个训练样本，每个样本是20*20像素的数字的灰度图像。
每个像素代表一个浮点数，表示该位置的灰度强度。
20×20的像素网格被展开成一个400维的向量。
X，每一个样本都变成了一行，这给了我们一个5000×400矩阵X，每一行都是一个手写数字图像的训练样本。
'''
data = scio.loadmat("ex3data1.mat")
X = data['X']  # 特征值
X = np.array([im.reshape((20, 20)).T for im in X])  # 把图像顺时间转90度
X = np.array([im.reshape(400) for im in X])  # 把图像翻转
y = data['y']  # 结果（0～9）
y = y.reshape((y.shape[0], 1))
print("X.shape:", X.shape)
print("y's shape:", y.shape)

# 单个图像
pick_one = np.random.randint(0, 5000)
plot_one_img(X[pick_one])
print('this should be', y[pick_one])

# 随机取100个样本
plot_100_img(X)



