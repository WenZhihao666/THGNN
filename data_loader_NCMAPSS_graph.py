import torch as tr
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from math import pi
units_index_train = ['2','5','10','16','18','20',]

units_index_test = ['11','14','15']

def load_array (sample_dir_path, unit_num, win_len, stride, sampling):
    filename =  'Unit%s_win%s_str%s_smp%s.npz' %(str(int(unit_num)), win_len, stride, sampling)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)

    return loaded['sample'].transpose(2, 0, 1), loaded['label']

def load_part_array_merge(npz_units):
    sample_array_lst = []
    label_array_lst = []
    for npz_unit in npz_units:
        loaded = np.load(npz_unit)
        sample_array_lst.append(loaded['sample'])
        label_array_lst.append(loaded['label'])
    sample_array = np.dstack(sample_array_lst)
    label_array = np.concatenate(label_array_lst)
    sample_array = sample_array.transpose(2, 0, 1)
    return sample_array, label_array


# idx=1,test_idx=3,feature_dimension=5,time_length=10,win_len=50
def NCMPDataIter_graph(sample_dir_path, idx, test_idx, feature_dimension, time_length, win_len=50, win_stride=1, sampling=10, sub=10):

    train_units_samples_lst =[]
    train_units_labels_lst = []
    test_units_samples_lst =[]
    test_units_labels_lst = []
    for index in units_index_train:
        # print("Load data index: ", index)
        sample_array, label_array = load_array (sample_dir_path, index, win_len, win_stride, sampling)
        sample_array = sample_array[idx::sub]
        label_array = label_array[idx::sub]
        # print("sub sample_array.shape", sample_array.shape)
        # print("sub label_array.shape", label_array.shape)
        train_units_samples_lst.append(sample_array)
        train_units_labels_lst.append(label_array)
    train_sample_array = np.concatenate(train_units_samples_lst)
    train_label_array = np.concatenate(train_units_labels_lst)

    if test_idx == 3:
        units_index_test_ = units_index_test
    else:
        units_index_test_ = [units_index_test[test_idx]]
    for index in units_index_test_:
        # print("Load data index: ", index)
        sample_array, label_array = load_array (sample_dir_path, index, win_len, win_stride, sampling)
        sample_array = sample_array[idx::sub]
        label_array = label_array[idx::sub]
        # print("sub sample_array.shape", sample_array.shape)
        # print("sub label_array.shape", label_array.shape)
        test_units_samples_lst.append(sample_array)
        test_units_labels_lst.append(label_array)
    test_sample_array = np.concatenate(test_units_samples_lst)
    test_label_array = np.concatenate(test_units_labels_lst)

    max_RUL = max(np.max(train_label_array), np.max(test_label_array))
    train_label_array = train_label_array/max_RUL
    test_label_array = test_label_array/max_RUL

    return train_sample_array,train_label_array,test_sample_array,test_label_array,max_RUL

def minmaxNor():
    sample_dir_path = './N-CMAPSS/Samples_whole/'
    rootPath = '/home/pengcheng/documents/datasets/HTGNN_datasets/NCMAPSSData/'
    # train_sample_array,train_label_array,test_sample_array,test_label_array,max_RUL = \
    #         NCMPDataIter_graph(sample_dir_path,1,3,5,10,50)
    time_length = 50
    minmax_scaler = MinMaxScaler()
    for test_idx in range(4):
        train_sample_array,train_label_array,test_sample_array,test_label_array,max_RUL = \
            NCMPDataIter_graph(sample_dir_path,idx=1,test_idx=test_idx,feature_dimension=5,time_length=10,win_len=50)
        print("train_sample_array.shape", train_sample_array.shape)
        print("test_sample_array.shape", test_sample_array.shape)
        for i in range(test_sample_array.shape[0]):
            test_sample_array[i] = minmax_scaler.fit_transform(test_sample_array[i])
        # np.savez(rootPath + 'test_samples/test_samples_{}_w{}_glgnn_norm.npz'.format(str(test_idx), str(time_length)), test_samples=test_sample_array)
        # np.savez(rootPath + 'test_samples/test_label_{}_w{}_glgnn.npz'.format(str(test_idx), str(time_length)), test_label=test_label_array)
    for i in range(train_sample_array.shape[0]):
            train_sample_array[i] = minmax_scaler.fit_transform(train_sample_array[i])
    # np.savez(rootPath + 'train_samples/train_samples_w{}_glgnn_norm.npz'.format(str(time_length)), train_samples=train_sample_array)
    # np.savez(rootPath + 'train_samples/train_label_w{}_glgnn.npz'.format(str(time_length)), train_label=train_label_array)

def compute_similarity_btw_nodes(nodei, nodej):
    maxv = 0.
    minv = 0.
    if max(nodei, nodej) <= 0:
        minv = np.abs(max(nodei, nodej))
        maxv = np.abs(min(nodei, nodej))
    else:
        if min(nodei, nodej) <= 0:
            maxv = max(nodei, nodej) + 2*np.abs(min(nodei, nodej))
            minv = np.abs(min(nodei, nodej))
        else:
            maxv = max(nodei, nodej)
            minv = min(nodei, nodej)
    dis = (maxv - minv)/(maxv + 1e-10)
    sim = np.cos(dis*(pi/2))
    return sim


def save_ele_adj_training(samples, time_length = 50):
    train_samples = samples
    train_adj_lst = []
    time_length = time_length
    # for i in range(10):
    for i in range(train_samples.shape[0]):
        if i % 5000 == 0:
            print(i)
        graph = np.transpose(train_samples[i])
        one_sample_adj_lst = []
        for tl in range(time_length):
            adj = np.zeros((graph.shape[0], graph.shape[0]))
            for nodei in range(graph.shape[0]):
                for nodej in range(nodei, graph.shape[0]):
                    sim = compute_similarity_btw_nodes(graph[nodei, tl], graph[nodej, tl])
                    adj[nodei, nodej] = sim
                    adj[nodej, nodei] = sim
            one_sample_adj_lst.append(adj)
        train_adj_lst.append(one_sample_adj_lst)
    train_adjs = np.asarray(train_adj_lst)
    np.savez('./N-CMAPSS/Samples_whole/train_adjs_w{}.npz'.format(str(time_length)), train_adjs=train_adjs)


def save_ele_adj_testing(samples, datasub, time_length = 50):
    train_samples = samples
    train_adj_lst = []
    time_length = time_length
    for i in range(train_samples.shape[0]):
        if i % 5000 == 0:
            print(i)
        graph = np.transpose(train_samples[i])
        one_sample_adj_lst = []
        for tl in range(time_length):
            adj = np.zeros((graph.shape[0], graph.shape[0]))
            for nodei in range(graph.shape[0]):
                for nodej in range(nodei, graph.shape[0]):
                    sim = compute_similarity_btw_nodes(graph[nodei, tl], graph[nodej, tl])
                    adj[nodei, nodej] = sim
                    adj[nodej, nodei] = sim
            one_sample_adj_lst.append(adj)
        train_adj_lst.append(one_sample_adj_lst)
    train_adjs = np.asarray(train_adj_lst)
    np.savez('./N-CMAPSS/Samples_whole/test_adjs_{}_w{}.npz'.format(str(datasub), str(time_length)), test_adjs=train_adjs)


if __name__ == '__main__':
    sample_dir_path = './N-CMAPSS/Samples_whole/'
    for test_idx in range(1):
        train_sample_array,train_label_array,test_sample_array,test_label_array,max_RUL = \
            NCMPDataIter_graph(sample_dir_path,idx=1,test_idx=test_idx,feature_dimension=1,time_length=50,win_len=50)
        test_sample_array = test_sample_array.reshape((-1, 50, 20))
        print('test_sample_array', test_sample_array.shape)
        # save_ele_adj_testing(test_sample_array, test_idx)
    train_sample_array = train_sample_array.reshape((-1, 50, 20))
    # save_ele_adj_training(train_sample_array)
    