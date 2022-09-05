#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import pandas as pd

import os
import sys
import argparse


class PlaceIDFilter:
    """
    用于通过标签进行分类
    """

    def __init__(self, df, keys=("BUILDING""ID", "FLOOR", "SPACE""ID")):
        self.number = 0
        self.df = df
        self.keys = keys

    def __call__(self, index):
        values = []
        for col in self.keys:
            values.append(str(self.df[col][index]))
        return "-".join(values)


def main_from_df(df):
    """

    Parameters
    ----------
    df
        待处理的数据

    Returns
    -------
    dict:
      {phoneID(int):
        {"placeID"(str):
          pd.DataFrame
         }
       }
    """
    # 先按照手机编号排序
    df_classification_dict = {a: b for a, b in df.groupby("PHONEID")}

    # 为每个切片按时间排序
    for df in df_classification_dict.values():
        df.sort_values("TIMESTAMP", inplace=True)

    # 将每个切片按标签再次切片
    items = df_classification_dict.items()
    df_classification_dict = {}
    for key, df in items:
        d = df_classification_dict[key] = {}
        pif = PlaceIDFilter(df)
        for place_id, df in df.groupby(pif):
            d.setdefault(place_id, []).append(df)

    # 将长度为1的列表转换为其值
    for phone_values in df_classification_dict.values():
        for k, v in phone_values.items():
            assert len(v) == 1
            phone_values[k] = v[0]

    return df_classification_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-DP', '--data_path',
                        type=str, default="../data/ujiindoorloc/",
                        help='寻找训练集、验证集文件和保存数据的路径')
    parser.add_argument('-TF', '--train_file',
                        type=str, default="training""data.csv",
                        help='训练集文件')
    parser.add_argument('-VF', '--validation_file',
                        type=str, default="validation""data.csv",
                        help='验证集文件')
    parser.add_argument('-SP', '--saving_path',
                        type=str, default="saved/" + os.path.splitext(os.path.basename(__file__))[0] + "/",
                        help='文件保存位置')

    args = parser.parse_args()

    data_path = args.data_path
    train_file = os.path.realpath(data_path + args.train_file)
    validation_file = os.path.realpath(data_path + args.validation_file)
    saving_path = os.path.realpath(data_path + args.saving_path)
