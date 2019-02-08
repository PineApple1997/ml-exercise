#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/2/7 22:30
# @Author  : qin yuxin
# @File    : ex1_multi.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
import seaborn as sns
from random import choice
sns.set(context="notebook", style="whitegrid")


def get_X(df):  # 读取特征
    """
    use concat to add intersect feature to avoid side effect
    not efficient for big dataset though
    """
    ones = pd.DataFrame(np.ones(len(df)), columns=['ones'])  # ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并(行是0)
    return data.iloc[:, :-1].values  # 这个操作返回 ndarray,不是矩阵


def get_y(df):  # 读取标签
    """ assume the last column is the target """
    return np.array(df.iloc[:, -1])  # df.iloc[:, -1]是指df的最后一列


def feature_normalize(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


def compute_cost(X, y, theta):
    m = X.shape[0]
    h = X @ theta
    J = sum((h-y)**2)/(2*m)
    return J


def gradient_descent(X, y, theta, alpha, iterations):
    m = X.shape[0]
    cost_data = [compute_cost(X, y, theta)]
    for i in range(iterations):
        theta = theta - (alpha / m) * (X.T @ (X @ theta - y))
        cost_data.append(compute_cost(X, y, theta))
    return theta, cost_data


df = pd.read_csv('ex1data2.txt', names=['size', 'room', 'price'])
df = feature_normalize(df)
X = get_X(df)
y = get_y(df)
X = X.reshape(X.shape[0], 3)
y = y.reshape(y.shape[0], 1)

alpha = 0.01
theta = np.zeros((X.shape[1], 1))    # X.shape[1]：特征数n
iterations = 1500

final_theta, cost_data = gradient_descent(X, y, theta, alpha, iterations)

print("final_theta:", final_theta)
f1 = plt.figure(1)
sns.lineplot(data=pd.DataFrame(cost_data), color=choice(sns.color_palette()))
plt.xlabel('iteration', fontsize=15)
plt.ylabel('cost', fontsize=15)
plt.show()

base = np.logspace(-4, -2, 3)  # base = 10^-5 ~ 10^-1 的5个等比元素
lr = np.sort(np.concatenate((base, base*3)))  # 得到10个learning rate
for alpha in lr:
    _, cost_data = gradient_descent(X, y, theta, alpha, iterations)
    sns.lineplot(data=pd.DataFrame(cost_data), color='r')

plt.legend(lr)
plt.show()



