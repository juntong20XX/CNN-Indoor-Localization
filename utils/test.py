#!/usr/bin/env python3

import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import callbacks

if __name__ == "__main__":
    # read train dataset
    # 读训练数据集
    train_x = np.load("my_training_array.npy")
    train_y = np.load("my_training_array_index.npy")

    train_x = train_x[::-1]
    train_y = train_y[::-1]

    # convert train_y to one-hot encoding
    # 转成one-hot编码
    tmp_train_y = np.zeros((train_y.shape[0], 8))
    for i in range(len(train_y)):
        tmp_train_y[i][int(train_y[i][0]) - 1] = 1
        tmp_train_y[i][int(train_y[i][1]) + 2] = 1
    train_y = tmp_train_y

    # reshape
    # 把[none, 52, 2] 变成 [none, 52, 2, 1]
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))

    # to fp32
    # 转 float
    train_x = train_x.astype('float32')
    train_y = train_y.astype('float32')

    # normalize
    train_x[::, 0, ::, ::] /= train_x[::, 0, ::, ::].max()
    train_x[::, 1, ::, ::] /= train_x[::, 1, ::, ::].min()

    # read val dataset
    # 读测试数据集
    val_x = np.load("my_validation_array.npy")
    val_y = np.load("my_validation_array_index.npy")

    val_x = val_x[::-1]
    val_y = val_y[::-1]

    # convert val_y to one-hot encoding
    # 转成one-hot编码
    tmp_val_y = np.zeros((val_y.shape[0], 8))
    for i in range(len(val_y)):
        tmp_val_y[i][int(val_y[i][0]) - 1] = 1
        tmp_val_y[i][int(val_y[i][1]) + 2] = 1
    val_y = tmp_val_y

    # reshape
    # 把[none, 52, 2] 变成 [none, 52, 2, 1]
    val_x = val_x.reshape((val_x.shape[0], val_x.shape[1], val_x.shape[2], 1))

    # to fp32
    # 转 float
    val_x = val_x.astype('float32')
    val_y = val_y.astype('float32')

    # normalize
    val_x[::, 0, ::, ::] /= val_x[::, 0, ::, ::].max()
    val_x[::, 1, ::, ::] /= val_x[::, 1, ::, ::].min()

    # model
    model = keras.models.load_model("model")

    predict_train = model.predict(train_x)

    acc_building = 0
    acc_floor = 0

    for i in range(len(train_y)):
        if predict_train[i, :3].argmax() + 1 == train_y[i, 0]:
            acc_building += 1
        # acc_building += sum(predict_train[i, :3].argmax() + 1 == train_y[i, 0])
        if predict_train[i, 3:].argmax() + 1 == train_y[i, 1]:
            acc_floor += 1
        # acc_floor += sum(predict_train[i, 3:].argmax() + 1 == train_y[i, 1])

    print("train building hit ratio: %.2f" % (acc_building / len(train_y)))
    print("train floor hit ratio %.2f" % (acc_floor / len(train_y)))

    predict_val = model.predict(val_x)

    acc_building = 0
    acc_floor = 0

    for i in range(len(val_y)):
        if predict_val[i, :3].argmax() + 1 == val_y[i, 0]:
            acc_building += 1
        if predict_val[i, 3:].argmax() + 1 == val_y[i, 1]:
            acc_floor += 1

    print("val building hit ratio: %.2f" % (acc_building / len(val_y)))
    print("val floor hit ratio %.2f" % (acc_floor / len(val_y)))