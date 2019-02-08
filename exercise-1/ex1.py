#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/1/15 21:11
# @Author  : qin yuxin
# @File    : ex1_single.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(context="notebook", style="whitegrid", palette="dark")
from mpl_toolkits.mplot3d import Axes3D


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
        print(compute_cost(X, y, theta))
        cost_data.append(compute_cost(X, y, theta))
    return theta, cost_data


# print("X.shape:", X.shape)
# print("y.shape:", y.shape)
# print("theta.shape:", theta.shape)


df = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
X = get_X(df)
y = get_y(df)
X = X.reshape(X.shape[0], 2)
y = y.reshape(y.shape[0], 1)

f1 = plt.figure(1)
#plt.plot(X, y, 'rx', 'MarkerSize', 10)  # Plot the data
sns.lmplot('population', 'profit', df, height=6, fit_reg=False)
plt.ylabel('Profit in $10,000s')  # Set the y−axis label
plt.xlabel('Population of City in 10,000s')  # Set the x−axis label

theta = np.zeros((X.shape[1], 1))  # initialize fitting parameters
iterations = 1500
alpha = 0.01
print(compute_cost(X, y, theta))
final_theta, cost_data = gradient_descent(X, y, theta, alpha, iterations)
print("theta:", final_theta)


predict1 = np.array([1, 3.5]) @ final_theta
predict2 = np.array([1, 7]) @ final_theta
print('For population = 35000, we predict a profit of \n', predict1[0]*10000)
print('For population = 70,000, we predict a profit of \n', predict2[0]*10000)

plt.plot(X[:, 1], X @ final_theta, '-', color='r')
plt.legend(['Linear regression', 'Training data'])
plt.show()

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]]).reshape((2, 1))
        J_vals[i, j] = compute_cost(X, y, t)


J_vals = J_vals.T
f2 = plt.figure(2)
ax = Axes3D(f2)

ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1, cmap='rainbow')
plt.xlabel('theta_0')
plt.ylabel('theta_1')

f3 = plt.figure(3)
plt.contourf(theta0_vals, theta1_vals, J_vals, 20, alpha=0.6)
plt.contour(theta0_vals, theta1_vals, J_vals)
plt.plot(final_theta[0], final_theta[1], 'r', marker='x', markerSize=10, LineWidth=2)
plt.show()

