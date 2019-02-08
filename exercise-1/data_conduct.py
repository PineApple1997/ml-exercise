#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/2/7 22:40
# @Author  : qin yuxin
# @File    : data_conduct.py
# @Software: PyCharm


import numpy as np
import pandas as pd


def get_X(df):  # 读取特征
    """
    use concat to add intersect feature to avoid side effect
    not efficient for big dataset though
    """
    ones = pd.DataFrame(np.ones(len(df)), columns=['ones'])  # ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并(行是0)
    return data.iloc[:, 0:2].values  # 这个操作返回 ndarray,不是矩阵


def get_y(df):  # 读取标签
    """ assume the last column is the target """
    return np.array(df.iloc[:, -1])  # df.iloc[:, -1]是指df的最后一列


def feature_normalize(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())
