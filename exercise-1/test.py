#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/2/6 21:52
# @Author  : qin yuxin
# @File    : test.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns


df = pd.read_csv('ex1data2.txt', names=['size', 'room', 'price'])
ones = pd.DataFrame(np.ones(len(df)), columns=['ones'])  # ones是m行1列的dataframe
data = pd.concat([ones, df], axis=1)
a = data.iloc[:, :-2].values

# sns.distplot(a)
sns.jointplot(x="x", y="y", data=a);
# x = np.arange(0, 3*np.pi, 0.1)
# y1 = np.sin(x)
# y2 = np.cos(x)
# f1 = plt.figure("figure1")
# plt.plot(x, y1, label="sin", color='r')
# plt.plot(x, y2, label="cos", color='b')
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin and cos")
plt.legend()
plt.show()