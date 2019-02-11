#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/2/10 10:10
# @Author  : qin yuxin
# @File    : ex2_reg.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ex2
import scipy.optimize as opt
from sklearn.metrics import classification_report
import seaborn as sns
sns.set(context="notebook", style="whitegrid")


def feature_mapping(x1, x2, power):
    data = {}  # dict类型
    for i in np.arange(power + 1):
        for p in np.arange(i + 1):
            data["f{}{}".format(i - p, p)] = (x1 ** (i-p)) * (x2 ** p)
    return pd.DataFrame(data)  # 返回df类型


def regularized_cost(theta, X, y, l=1):
    theta = theta.reshape(-1, 1)
    _theta = theta[1:]
    regularized_term = (l / (2 * len(X))) * np.sum(_theta ** 2)
    return ex2.compute_cost(theta, X, y) + regularized_term


def regularized_gradient(theta, X, y, l=1):
    theta = theta.reshape(-1, 1)
    regularized_term = l / len(X) * theta
    regularized_term[0] = 0
    return ex2.get_gradient(theta, X, y) + regularized_term


''' 读取数据 返回df类型 '''
df = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'quality'])

''' 画出原始数据的散点图 '''
f1 = plt.figure(1)
sns.lmplot('test1', 'test2', df, hue='quality', height=7, markers=['o', '+'], fit_reg=False)
plt.xlabel("test1")
plt.ylabel("test2")
plt.title("raw data")
plt.show()


''' feature mapping '''
X = feature_mapping(df.test1.values, df.test2.values, 6)
y = df.quality.values.reshape((X.shape[0], 1))
theta = np.zeros((X.shape[1], 1))
# print("X.shape:", X.shape)
# print("y.shape:", y.shape)
# print("theta.shape:", theta.shape)

''' 正则化 '''
print("regularized cost:", regularized_cost(theta, X, y, 1))
print("regularized gradient", (regularized_gradient(theta, X, y, 1)))

''' 拟合参数 '''
res = opt.fmin_tnc(func=regularized_cost, x0=theta, fprime=regularized_gradient, args=(X, y, 1))
final_theta = res[0]
print("final_theta:", final_theta)
y_pred = ex2.predict(X, final_theta)
print(classification_report(y, y_pred))


'''画决策边界'''
t1 = np.linspace(-1, 1.5, 1000)
t2 = np.linspace(-1, 1.5, 1000)
coordinates = [(x, y) for x in t1 for y in t2]
x_cord, y_cord = zip(*coordinates)  # 解压 返回二维矩阵
mapped_cord = feature_mapping(x_cord, y_cord, 6)  # 给坐标上的x y也做feature mapping
inner_product = mapped_cord.values @ final_theta  # 内积 假设函数

decision = mapped_cord[np.abs(inner_product) < (2 * 10**-3)]
print("decision:", decision)
x1 = decision.f10  # test1
x2 = decision.f01  # test2
f2 = plt.figure(2)
sns.lmplot('test1', 'test2', df, hue='quality', height=7, markers=['o', '+'], fit_reg=False)
plt.scatter(x1, x2, color='R', s=10)
plt.xlabel("test1")
plt.ylabel("test2")
plt.title("Decision Boundary")
plt.show()