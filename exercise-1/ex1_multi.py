#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/2/7 22:30
# @Author  : qin yuxin
# @File    : ex1_multi.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid")
import data_conduct
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('ex1data2.txt', names=['size', 'room', 'price'])

df = data_conduct.feature_normalize(df)
X = data_conduct.get_X(df)
y = data_conduct.get_y(df)
X = X.reshape(X.shape[0], 3)
y = y.reshape(y.shape[0], 1)

f1 = plt.figure(1)
ax = Axes3D(f1)
ax.scatter(X[:, 1], X[:, 2], y[:, 0], c='r')  # 绘制数据点

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

alpha = 0.01
theta = np.zeros((X.shape[1], 1))    # X.shape[1]：特征数n
iterations = 1500

final_theta, cost_data = data_conduct.gradient_descent(X, y, theta, alpha, iterations)

print("final_theta:", final_theta)
#
# plt.plot(X[:, 1], X @ final_theta, '-', color='r')
# ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1, cmap='rainbow')
# plt.legend(['Linear regression', 'Training data'])


f2 = plt.figure(2)
sns.lineplot(data=pd.DataFrame(cost_data), color="b")
plt.xlabel('iteration', fontsize=15)
plt.ylabel('cost', fontsize=15)


base = np.logspace(-4, -2, 3)  # base = 10^-5 ~ 10^-1 的5个等比元素
lr = np.sort(np.concatenate((base, base*3)))  # 得到10个learning rate
for alpha in lr:
    _, cost_data = data_conduct.gradient_descent(X, y, theta, alpha, iterations)
    sns.lineplot(data=pd.DataFrame(cost_data), color='r')
plt.legend(lr)
plt.show()


