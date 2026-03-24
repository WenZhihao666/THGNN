import os
import csv
import random
import numpy as np
from numpy.core.fromnumeric import transpose
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import interpolate
import math

import torch.utils.data as data
import logging
# from config import config
import data_loader_CMPS_original


def resize_graph(data, time_length, feature_dimension):
    bs, _, nodes = np.shape(data)
    data = np.reshape(data, [bs, time_length, feature_dimension, nodes])
    # print(data.shape)
    data = np.transpose(data, [0,1,3,2])
    return data


def CMPDataIter_graph(sample_dir_path,idx, feature_dimension, time_length, max_RUL = 125, net_name = 1):
    data_iter = data_loader_CMPS_original.CMPDataIter(data_root=sample_dir_path,
                                                    data_set=idx,
                                                    max_rul=max_RUL,
                                                    seq_len=feature_dimension * time_length)

    train_x = data_iter.out_x
    train_x = resize_graph(train_x, time_length, feature_dimension)
    train_y = data_iter.out_y
    # print(data_iter)
    test_x = data_iter.test_x
    test_x = resize_graph(test_x, time_length, feature_dimension)
    test_y = data_iter.test_y

    return train_x, train_y, test_x, test_y, max_RUL

if __name__ == '__main__':
    CMPDataIter_graph('./CMAPSS/', 'FD001', 1, 30)