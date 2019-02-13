#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/2/9 13:04
# @Author  : qin yuxin
# @File    : ex2.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from exercise_1_linear_regression import ex1_data_conduct
import scipy.optimize as opt
from sklearn.metrics import classification_report
import seaborn as sns
sns.set(context="notebook", style="whitegrid")


def sigmoid(z):
    return 1/(1+np.exp(-z))


def compute_cost(theta, X, y):
    theta = theta.reshape((X.shape[1], 1))
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))


def get_gradient(theta, X, y):
    theta = theta.reshape((X.shape[1], 1))
    return 1/len(X) * X.T @ (sigmoid(X @ theta) - y)


def predict(X, theta):
    probability = sigmoid(X @ theta)
    return [1 if x >= 0.5 else 0 for x in probability]

def main():
    df = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admission'])
    X = ex1_data_conduct.get_X(df)  # x's shape: 100*3 (1, exam1, exam2)
    y = ex1_data_conduct.get_y(df)  # y's shape: 100*1
    X = X.reshape(X.shape[0], 3)
    y = y.reshape(y.shape[0], 1)

    f1 = plt.figure(1)
    sns.lmplot('exam1', 'exam2', data=df, height=6, fit_reg=False, hue='admission', markers=["o", "+"])
    plt.xlabel('exam1 score')  # Set the y−axis label
    plt.ylabel('exam2 score')  # Set the x−axis label
    plt.title("raw data")
    plt.show()

    f2 = plt.figure(2)
    sns.lineplot(np.arange(-10, 10, 0.01), sigmoid(np.arange(-10, 10, 0.01)))
    plt.title("sigmoid figure")
    plt.xlabel("z")
    plt.ylabel("g(z)")
    plt.show()

    theta = np.zeros((X.shape[1], 1))
    print("cost:", compute_cost(theta, X, y))
    print("gradient:", get_gradient(theta, X, y))
    print("gradient's shape", get_gradient(theta, X, y).shape)

    # 用scipy库的optimize来自动拟合参数theta
    result = opt.fmin_tnc(func=compute_cost, x0=theta, fprime=get_gradient, args=(X, y))
    final_theta = result[0]
    print("final_theta:", final_theta)
    y_pred = predict(X, final_theta)
    print(classification_report(y, y_pred))  # 为啥micro和macro值不同？ not fixed

    # theta0 + theta1 * x1 + theta2 * x2 = 0
    # x2 = (-theta0 - theta1 * x1) / theta2
    x1 = np.arange(25, 100, step=0.1)
    x2 = -(final_theta[0] + x1*final_theta[1]) / final_theta[2]

    f3 = plt.figure(3)
    sns.lmplot('exam1', 'exam2', data=df, height=6, fit_reg=False, hue='admission', markers=["o", "+"])
    sns.lineplot(x1, x2, color="g")
    plt.xlabel('exam1 score')  # Set the y−axis label
    plt.ylabel('exam2 score')  # Set the x−axis label
    plt.title("Data")
    plt.show()


if __name__ == "__main__":
    main()