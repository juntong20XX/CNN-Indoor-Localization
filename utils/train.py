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

    # new model
    # 创建模型
    model = keras.Sequential()

    # # add conv
    # # 卷积层
    model.add(layers.Conv2D(128, kernel_size=(2, 1), activation='relu', input_shape=(2, 51, 1)))

    # model.add(layers.Conv1D(64, kernel_size=4, activation='relu'))

    # dropout
    # 随机断开连接
    model.add(layers.Dropout(0.25, name="droupout-0"))

    # flat
    # 展平
    model.add(layers.Flatten())

    # classifier
    # 分类器
    classifier_hidden_layers = [64, 32]
    for i in range(len(classifier_hidden_layers)):
        model.add(layers.Dense(classifier_hidden_layers[i], name="classifier-hidden-" + str(i), activation='relu'))
        model.add(layers.Dropout(0.25, name="droupout-" + str(i + 1)))

    model.add(layers.Dense(8, name="activation-0", activation='sigmoid'))  # 'sigmoid' for multi-label classification

    # summary
    # 展示模型
    print(model.summary())

    # compile
    # 编译模型
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

    # train the model
    # 训练模型
    callback = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)
    model.fit(train_x, train_y, batch_size=1, epochs=100, verbose=1, validation_data=(val_x, val_y), callbacks=[callback])

    # save the model for later use
    # 保存模型
    model.save("model")

    predict_train = model.predict(train_x)

    acc_building = 0
    acc_floor = 0

    for i in range(len(train_y)):
        if predict_train[i, :3].argmax() + 1 == train_y[i, 0]:
            acc_building += 1
        if predict_train[i, 3:].argmax() + 1 == train_y[i, 1]:
            acc_floor += 1

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
