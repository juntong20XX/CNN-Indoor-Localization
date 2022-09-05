# CNN Indoor Localization

A prototype indoor localization system based on CNN. This repository is for the implementation of CNN models for scalable indoor localization with Wi-Fi fingerprinting.

## Brief

For large-scale multi-building and multi-layer indoor localization based on a single dataset of Wi-Fi received signal strength (RSS), we propose a scalable representation of neural network input data that reduces the number of hyperparameters, which makes battery-powered mobile devices and embedded systems as a target platform. Based on the UJIIndoorLoc dataset, we use an RSSI-based ranking and (CNN) to build a multi-label indoor localization model. We also study the relationship between localization accuracy and the number of input signals and verified that the smaller input dimension has limited effect on the model accuracy.

**TODO: 添加关于SURF的内容**

## Data

The data used in this project is `UJIIndoorLoc`.

## PYPI Requisites

You can see the packages in file [utils/requisites.txt](./utils/requisites.txt).

- `numpy`
- `pandas`
- `tables`
- `tensorflow`

## Running and Code

First run `rssi_representation.py` to generate files for training and validation.

Second run `train*.py` (all the three files are ok) to create CNN model.

Now, you can run `test.py` to test the model, you may need to change the model and validation path.



## Result

## Authors
