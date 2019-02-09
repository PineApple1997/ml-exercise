#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/2/9 13:04
# @Author  : qin yuxin
# @File    : ex2.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid")
from mpl_toolkits.mplot3d import Axes3D
import data_conduct
import scipy.optimize as opt


def sigmoid(z):
    return 1/(1+np.exp(-z))


def compute_cost(X, y, theta):
    print("theta's shape in cost:", theta.shape)
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))


def get_gradient(X, y, theta):
    print("theta's shape in gradient:", theta.shape)
    return 1/len(X) * X.T @ (sigmoid(X @ theta) - y)


df = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admission'])
X = data_conduct.get_X(df)  # x's shape: 100*3 (1, exam1, exam2)
y = data_conduct.get_y(df)  # y's shape: 100*1
X = X.reshape(X.shape[0], 3)
y = y.reshape(y.shape[0], 1)


f1 = plt.figure(1)
sns.lmplot('exam1', 'exam2', data=df, height=6, fit_reg=False, hue='admission', markers=["o", "+"])
plt.ylabel('exam1 score')  # Set the y−axis label
plt.xlabel('exam2 score')  # Set the x−axis label
plt.title("raw data")
plt.show()

f2 = plt.figure(2)
sns.lineplot(np.arange(-10, 10, 0.01), sigmoid(np.arange(-10, 10, 0.01)))
plt.title("sigmoid figure")
plt.xlabel("z")
plt.ylabel("g(z)")
plt.show()

theta = np.zeros((X.shape[1], 1))


print("theta's shape:", theta.shape)
print("cost:", compute_cost(X, y, theta))
print("gradient", get_gradient(X, y, theta))
print("gradient's shape", get_gradient(X, y, theta).shape)

# 用scipy库的optimize来自动拟合参数theta
print("theta's shape:", theta.shape)
res = opt.minimize(fun=compute_cost, x0=theta, args=(X, y), method='Newton-CG', jac=get_gradient)
print(res)


