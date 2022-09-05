#!/usr/bin/env python3

import numpy as np
import pandas as pd
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import callbacks

PATHs = {
    "train-h5": "data/uji""indoor""loc/saved/training_df.h5",
    "validation-h5": "data/uji""indoor""loc/saved/validation_df.h5",
    "model": "model",
}


def load_hdf_dataset(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    load the .h5 dataset from `path`.
    Parameters
    ----------
    path
        the path of .h5 file
    Returns
    -------
    df["WAPs_RSSIs"], df["BUILDING""ID"], df["FLOOR"]
    """
    df = pd.read_hdf(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("need a dataframe, got", type(df))
    return (df["WAPs_RSSIs"].to_numpy(),
            df["BUILDING""ID"].to_numpy(),
            df["FLOOR"].to_numpy(),
            )


def get_no_signal_array(first_two_shape: tuple[int, int], no_signal_value: int = -110) -> np.ndarray:
    """
    Return an array as no signal data.

    Parameters
    ----------
    first_two_shape
    no_signal_value

    Returns
    -------
    np.ndarray, like [[[0, no_signal_value], [0, no_signal_value], ...],
                      [[0, no_signal_value], [0, no_signal_value], ...],
                                           ...
                      [[0, no_signal_value], [0, no_signal_value], ...]]
    array shape is (*front_two_shape, 2)
    """
    ret_1 = np.zeros(first_two_shape, dtype=int)
    ret_2 = np.full(first_two_shape, no_signal_value)

    return np.stack((ret_1, ret_2), 2)


def copy_data_for_model(data: np.ndarray, disk: np.ndarray):
    """
    copy data to no_signal_array (parameter 'disk') for model
    Parameters
    ----------
    data
    disk

    Returns
    -------
    None
    """
    for index, line in enumerate(data):
        disk_line = disk[index, :len(line)]
        disk_line[:, 0] = line[:, 0] + 1  # 这里+1是因为无信号是0, 剩下的向后顺延
        disk_line[:, 1] = line[:, 1]


def get_one_hot_array(building: np.ndarray, floor: np.ndarray, return_width=8) -> np.ndarray:
    lb = len(building)
    lf = len(floor)
    assert lb == lf, ValueError("length of `build` and `floor` should be same, got %d and %d" % (lb, lf))
    ret = np.zeros((lb, return_width))
    for idx in range(lb):
        ret[idx][int(building[idx])] = 1
        ret[idx][int(floor[idx]) + 3] = 1
    return ret


if __name__ == "__main__":
    import getpass

    if getpass.getuser() in ("Jessi", "john"):
        PATHs["train-h5"] = "../" + PATHs["train-h5"]
        PATHs["validation-h5"] = "../" + PATHs["validation-h5"]
        PATHs["model"] = "../models/" + PATHs["model"]

    # read train dataset
    # 读训练数据集
    train_waps_rssis, train_building, train_floor = load_hdf_dataset(PATHs["train-h5"])

    # create an array full of no signal WI-FI for later use
    # 创建一个全是无信号的train_x
    train_x: np.ndarray = get_no_signal_array((len(train_waps_rssis), 51))

    # copy data
    # 复制数据
    copy_data_for_model(train_waps_rssis, train_x)

    # convert train_y to one-hot encoding
    # 转成one-hot编码
    # 创建全是0的train_y
    train_y = get_one_hot_array(train_building, train_floor)

    # reshape from [none, 52, 2] to [none, 52, 2, 1]
    # 把[none, 52, 2] 变成 [none, 52, 2, 1]
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))

    # to fp32
    # 转 float
    train_x = train_x.astype('float32')
    train_y = train_y.astype('float32')

    # normalize
    train_x[::, ::, 0, ::] /= 521
    train_x[::, ::, 1, ::] += 110
    train_x[::, ::, 1, ::] /= 110

    # # read val dataset
    # # 读测试数据集
    val_waps_rssis, val_building, val_floor = load_hdf_dataset(PATHs["validation-h5"])

    # create an array full of no signal WI-FI for later use
    # 创建一个全是无信号的val_x
    val_x: np.ndarray = get_no_signal_array((len(val_waps_rssis), 51))

    # copy data
    # 复制数据
    copy_data_for_model(val_waps_rssis, val_x)

    # convert val_y to one-hot encoding
    # 转成one-hot编码
    # 创建全是0的val_y
    val_y = get_one_hot_array(val_building, val_floor)

    # reshape from [none, 52, 2] to [none, 52, 2, 1]
    # 把[none, 52, 2] 变成 [none, 52, 2, 1]
    val_x = val_x.reshape((val_x.shape[0], val_x.shape[1], val_x.shape[2], 1))

    # to fp32
    # 转 float
    val_x = val_x.astype('float32')
    val_y = val_y.astype('float32')

    # normalize
    val_x[::, ::, 0, ::] /= 521
    val_x[::, ::, 1, ::] += 110
    val_x[::, ::, 1, ::] /= 110

    # new model
    # 创建模型
    model = keras.Sequential()

    # add conv
    # 卷积层
    model.add(layers.Conv2D(16, kernel_size=(2, 1), activation='relu',
                            input_shape=(train_x.shape[1], train_x.shape[2], train_x.shape[3])))

    model.add(layers.Conv2D(16, kernel_size=(2, 1), activation='relu'))

    # dropout
    # 随机断开连接
    model.add(layers.Dropout(0.25, name="droupout-0"))

    # flat
    # 展平
    model.add(layers.Flatten())

    # classifier
    # 分类器
    classifier_hidden_layers = [64, 128]
    for i in range(len(classifier_hidden_layers)):
        model.add(layers.Dense(classifier_hidden_layers[i], name="classifier-hidden-" + str(i), activation='relu'))
        model.add(layers.Dropout(0.25, name="droupout-" + str(i + 1)))

    model.add(layers.Dense(8, name="activation-0", activation='sigmoid'))  # 'sigmoid' for multi-label classification

    # summary
    # 展示模型
    model.summary()

    # compile
    # 编译模型
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    # train the model
    # 训练模型
    callback = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
    model.fit(train_x, train_y, batch_size=4, epochs=100, verbose=1, validation_data=(val_x, val_y),
              callbacks=[callback])
    # save the model for later use
    # 保存模型
    model.save(PATHs["model"])

    # 使用网络预测训练集
    predict_train = model.predict(train_x)

    # 正确率变量
    acc_building = 0
    acc_floor = 0

    # 循环判断正确
    for i in range(len(train_y)):
        if predict_train[i, :3].argmax() == train_y[i, :3].argmax():
            acc_building += 1
        if predict_train[i, 3:].argmax() == train_y[i, 3:].argmax():
            acc_floor += 1

    print("train building hit ratio: %.2f" % (acc_building / len(train_y)))
    print("train floor hit ratio %.2f" % (acc_floor / len(train_y)))

    # 使用网络预测测试集
    predict_val = model.predict(val_x)

    # 正确率变量
    acc_building = 0
    acc_floor = 0

    # 循环判断正确
    for i in range(len(val_y)):
        if predict_val[i, :3].argmax() == val_y[i, :3].argmax():
            acc_building += 1
        if predict_val[i, 3:].argmax() == val_y[i, 3:].argmax():
            acc_floor += 1

    print("val building hit ratio: %.2f" % (acc_building / len(val_y)))
    print("val floor hit ratio %.2f" % (acc_floor / len(val_y)))
