#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file     rssi_representation.py
@author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
@date     2022-01-27

@brief    Time series representation of RSSIs for DNN-based
          large-scale indoor localization
"""

# import cloudpickle
import os
import typing
import itertools

import numpy as np
import pandas as pd


def map_rssi(x: list[list[int]]) -> typing.Optional[np.array]:
    """
    返回一个二维列表, 内容为 x 中不等于100的值的下标和这个非100值;
    若没有不等于100的值, 则返回 np.nan.

    return a 2D array with [WAP_index, RSSI] if RSSI is not 100, return np.nan if len(array) is 0.

    Notes
    -----
    WAP index starts with 1

    Parameters
    ----------
    x
        待处理的列表

    Returns
    -------
    np.array OR np.nan
    """
    # y = list(zip(range(len(x)), x))
    # y = list(enumerate(x))  # 这是原先的
    y = list(enumerate(x, 1))  # 这是修改的，改为从1开始数
    y = np.array([a for a in y if a[1] != 100])  # Why a[1] != 100 ? Note: a[1]是x的元素
    if len(y) > 0:
        # 根据y第二列排序, 别忘了第一列是初始下标
        return y[np.argsort(y[:, 1])][::-1]
    else:
        return np.nan


def ts_length(x: np.ndarray | None):
    """
    返回输入的形状, 若类型不是np.ndarray, 则返回0
    return x.shape[0] if x is ndarray else 0

    Parameters
    ----------
    x
        np.ndarray or np.nan
    Returns
    -------

    """
    if type(x) == np.ndarray:  # isinstance(x, np.ndarray) 其实这里写 type(x) is np.ndarray 也可以
        return x.shape[0]
    else:
        return 0


def get_place_id(line):
    return "%s-%s-%s" % tuple(line)


def clean_data(df: pd.DataFrame):
    """
    数据清洗操作, 是老师对 training_data 和 validation_data 操作的集合.

    Notes
    ----------
    此操作直接对修改 df

    Parameters
    ----------
    df
        pd.DataFrame, 加载自数据源

    Returns
    -------
    df

    """
    # wap_columns 是所有WAP的列名
    wap_columns = itertools.takewhile(lambda x: x.startswith("WAP"), df.columns)
    wap_columns = list(wap_columns)
    # 新建一列名为"RSSIs"的数据, 该列每格内容对应着这一行所有的WAP的值列表
    df["RSSIs"] = df.loc[:, wap_columns].values.tolist()  # df.loc 切片第一个值表示纵向,选择“行”; 第二个表示横向, 选择“列”
    # 新建一列名为"WAPs_RSSIs"的数据, 这一列数据每格是一个二维列表 或nan
    # 是列表时, 该列表的第二列代表这一行的WAP中非100的元素, 第一列是其对应的下标
    # 当此行所有WAP数据是100时, 此格内容为 np.nan
    df["WAPs_RSSIs"] = df["RSSIs"].apply(map_rssi)
    # 新建一列, 这一列数据每格是一个二维列表 或nan
    df["TS_LENGTH"] = df["WAPs_RSSIs"].apply(ts_length)
    # 删除WAP全部是100的行
    df.drop(df[df.TS_LENGTH <= 0].index, inplace=True)  # 不过好像长度不会小于0吧？
    # Juntong-add
    # df.sort_values("TIMESTAMP", inplace=True)  # 经尝试，使用此排序的结果与未使用一致
    # df.sort_values("TS_LENGTH", inplace=True)
    df["PLACE_ID"] = df[["BUILDING""ID", "FLOOR", "SPACE""ID"]].apply(get_place_id, axis="columns")
    # Juntong-end
    return df


# training data
df_fname = '../data/ujiindoorloc/saved/training_df.h5'
if os.path.isfile(df_fname) and (os.path.getmtime(df_fname) >
                                 os.path.getmtime(__file__)):
    training_df = pd.read_hdf(df_fname)
else:
    training_df = pd.read_csv('../data/ujiindoorloc/training''data.csv', header=0)
    clean_data(training_df)
    training_df.to_hdf(df_fname, key='training_df')

# validation data
df_fname = '../data/ujiindoorloc/saved/validation_df.h5'
if os.path.isfile(df_fname) and (os.path.getmtime(df_fname) >
                                 os.path.getmtime(__file__)):
    validation_df = pd.read_hdf(df_fname)
else:
    validation_df = pd.read_csv('../data/ujiindoorloc/validation''data.csv', header=0)
    clean_data(validation_df)
    validation_df.to_hdf(df_fname, key='validation_df')


def print_data(df: pd.DataFrame):
    """
    打印数据，源于老师的代码。
    
    Parameters
    ----------
    df
        待显示的DataFrame对象
    Returns
    -------
    None
    """
    print(df.head())
    df["TS_LENGTH"].describe()
    print(f'- Average number of RSSIs: {df["TS_LENGTH"].mean():e}')
    print(f'- Maximum number of RSSIs: {df["TS_LENGTH"].max():e}')
    print(f'- Minimum number of RSSIs: {df["TS_LENGTH"].min():e}')
    print(f'- Number of elements without RSSIs: {df["WAPs_RSSIs"].isna().sum():d}')


# summary of new rssi dataframes
print("Training data:")
print_data(training_df)

print("Validation data:")
print_data(validation_df)


def data_wap_rssi(df: pd.DataFrame):
    """
    将df中的`WAPs_RSSIs`抽出，组成大的DataFrame
    Parameters
    ----------
    df
        待处理的数据集
    Returns
    -------

    """

    df = df.sort_values("TS_LENGTH")
    # length = df["TS_LENGTH"].max()
    length = 51  # 51 是训练集和测试集的最大长度
    ret_width = length * len(df.index)

    # ---------- 此处是使用顺延WAP序号进行补全 ----------
    # sta = 520  # 是不存在的WAP索引开始处
    # wap_s = np.arange(sta, sta + ret_width)
    # rssi_s = np.zeros(ret_width, dtype=int)
    wap_s = np.zeros(ret_width, dtype=int)
    rssi_s = np.full(ret_width, -110, dtype=int)

    ret = np.vstack((wap_s, rssi_s))

    # ---------- 这里是用0补全 ----------
    # ret = np.zeros((2, ret_width))

    index = 0
    for ls in df["WAPs_RSSIs"]:
        i = length
        for w, r in ls:
            ret[0][index] = w
            ret[1][index] = r

            i -= 1
            index += 1
        index += i

    return list(df["PLACE_ID"]), ret


training_npy_path = "../data/ujiindoorloc/saved/my_training_array.npy"
training_json_path = os.path.splitext(training_npy_path)[0] + "_index.json"
validation_npy_path = "../data/ujiindoorloc/saved/my_validation_array.npy"
validation_json_path = os.path.splitext(validation_npy_path)[0] + "_index.json"


def array_to_3d(arr: np.ndarray, length: int) -> np.ndarray:
    """

    Parameters
    ----------
    arr
        2D array with [[WAP_index, RSSI], ...]
    length
         = len(place_id_list)

    Returns
    -------
    np.ndarray which is 3D

    """

    length = arr.shape[1] // length
    ls = [arr[:, i*length:(i+1)*length] for i in range(arr.shape[1] // length)]
    return np.array(ls)


if os.path.isfile(training_npy_path) and (os.path.getmtime(training_npy_path) > os.path.getmtime(__file__)):
    # load data
    training_array = np.load(training_npy_path)
    np.load(os.path.splitext(training_json_path)[0] + ".npy")
else:
    training_place_id_list, training_array = data_wap_rssi(training_df)
    training_array = array_to_3d(training_array, len(training_place_id_list))
    np.save(training_npy_path, training_array)
    np.save(os.path.splitext(training_json_path)[0], np.array([i.split("-") for i in training_place_id_list]))

if os.path.isfile(validation_npy_path) and (os.path.getmtime(validation_npy_path) > os.path.getmtime(__file__)):
    validation_array = np.load(validation_npy_path)
    np.load(os.path.splitext(validation_json_path)[0] + ".npy")
else:
    validation_place_id_list, validation_array = data_wap_rssi(validation_df)
    validation_array = array_to_3d(validation_array, len(validation_place_id_list))
    np.save(validation_npy_path, validation_array)
    np.save(os.path.splitext(validation_json_path)[0], np.array([i.split("-") for i in validation_place_id_list]))
