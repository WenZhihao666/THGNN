import torch as tr
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os
import thgnn
from data_loader_CMAPSS_graph import CMPDataIter_graph
from data_loader_NCMAPSS_graph import NCMPDataIter_graph
import argparse
import matplotlib.pyplot as plt
import random
import sys
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import warnings

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

seed_lst = [1, 4, 16, 64, 128]


def setup_seed(seed):
    tr.manual_seed(seed)
    tr.cuda.manual_seed(seed)
    tr.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    tr.backends.cudnn.deterministic = True


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data, label, corr, heter_types):
        super(Load_Dataset, self).__init__()

        self.data = data
        self.label = label
        self.corr = corr
        self.heter_types = heter_types
        self.len = data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.corr[index], self.heter_types

    def __len__(self):
        return self.len


class Train():
    def __init__(self, args):
        self.args = args
        if args.data_name == 'CMPS':
            self.train_data, self.train_label, self.test_data, self.test_label, self.max_RUL = CMPDataIter_graph(
                args.data_path, 'FD00{}'.format(str(args.data_sub)), args.feature_dimension, args.time_length)
        elif args.data_name == 'NCMPS':
            self.train_data, self.train_label, self.test_data, self.test_label, self.max_RUL = NCMPDataIter_graph(
                args.data_path, args.sub_idx, args.data_sub, args.feature_dimension, args.time_length)
        self.cor_coeff_train_value = self.corrcoef_generation_full(training=True)
        self.cor_coeff_test_value = self.corrcoef_generation_full(training=False)
        # self.heter_types = self.heter_types_generation()
        self.heter_types = self.types_generation()

        self.train_data = self.cuda_(self.train_data)
        self.train_label = self.cuda_(self.train_label)
        self.train_label = tr.squeeze(self.train_label)

        # print('self.train_label', self.train_label[-1000:])
        # max_value_train, max_index = tr.max(self.train_label, dim=0)
        # min_value_train, max_index = tr.min(self.train_label, dim=0)

        self.test_data = self.cuda_(self.test_data)
        self.test_label = self.cuda_(self.test_label)
        self.test_label = tr.squeeze(self.test_label)

        # print('self.test_label', self.test_label[-1000:])
        # max_value_test, max_index = tr.max(self.test_label, dim=0)
        # min_value_test, max_index = tr.min(self.test_label, dim=0)

        self.cor_coeff_train_value = self.cuda_(self.cor_coeff_train_value)
        self.cor_coeff_test_value = self.cuda_(self.cor_coeff_test_value)
        self.heter_types = self.cuda_(self.heter_types)

        train_data_loader = Load_Dataset(self.train_data, self.train_label, self.cor_coeff_train_value,
                                         self.heter_types)
        self.train_loader = tr.utils.data.DataLoader(dataset=train_data_loader, batch_size=args.batch_size,
                                                     shuffle=True, drop_last=False,
                                                     num_workers=0)

        test_data_loader = Load_Dataset(self.test_data, self.test_label, self.cor_coeff_test_value, self.heter_types)
        self.test_loader = tr.utils.data.DataLoader(dataset=test_data_loader, batch_size=args.batch_size,
                                                    shuffle=False, drop_last=False,
                                                    num_workers=0)
        self.net = thgnn.HTGNNModel(self.args).to(tr.device(self.args.device))

        # for name, param in self.net.named_parameters():
        #     print(f"Layer: {name}, Parameter Shape: {param.shape}")

        # for abaltion test
        # self.abla_net = HTGNN.GNNModel(self.args).to(tr.device(self.args.device))
        # self.net = self.abla_net
        self.loss_function = nn.MSELoss()
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.l2reg)  # 0.0001
        # self.lr_schedular = optim.lr_scheduler.MultiStepLR(self.optim, [5,10,20,25],0.5)
        self.lr_schedular = optim.lr_scheduler.MultiStepLR(self.optim, [10, 20, 30, 40], 0.5)

    def heter_types_generation(self):
        name = self.args.data_name
        if name == 'CMPS':
            lst = [0, 0, 1, 2, 3, 3, 2, 4, 3, 3, 5, 6, 7, 7]
        elif name == 'NCMPS':
            pass
        combTyps = []
        adj = np.zeros((len(lst), len(lst)))
        for i in range(len(adj)):
            for j in range(i + 1, len(adj)):
                comb = tuple(sorted((lst[i], lst[j])))
                tyId = -1
                if comb in combTyps:
                    tyId = combTyps.index(comb)
                else:
                    combTyps.append(comb)
                    tyId = len(combTyps) - 1
                adj[i, j] = tyId
                adj[j, i] = tyId
        return adj

    def types_generation(self):
        name = self.args.data_name
        if name == 'CMPS':
            node_types = np.array([0, 0, 1, 2, 3, 3, 2, 4, 3, 3, 5, 6, 7, 7])
        elif name == 'NCMPS':
            node_types = np.array([0, 1, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 3, 4])
        return node_types

    def corrcoef_generation_full(self, training=True):
        loadFixed = False
        if loadFixed:
            name = self.args.data_name
            path = self.args.data_path
            subid = self.args.data_sub
            if name == 'CMPS':
                if training:
                    adjs = \
                        np.load(path + 'CMAPSSData/train_adjs_FD00{}.npz'.format(str(subid)))['train_adjs']
                else:
                    adjs = \
                        np.load(path + 'CMAPSSData/test_adjs_FD00{}.npz'.format(str(subid)))['test_adjs']
            elif name == 'NCMPS':
                if training:
                    adjs = \
                        np.load(path + '/train_adjs_w50.npz')['train_adjs']
                else:
                    adjs = \
                        np.load(path + '/test_adjs_{}_w50.npz'.format(str(subid)))['test_adjs']
        else:
            adjs = np.zeros(self.train_data.shape[0])
        return adjs

    def Train_batch(self):
        self.net.train()
        loss_ = 0
        for data, label, bs_corrcoef, heter_types in self.train_loader:
            prediction, reg_cost = self.net(data, bs_corrcoef, heter_types)
            prediction = tr.squeeze(prediction)

            loss = self.loss_function(prediction, label)
            if self.args.film_reg == 0:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            else:
                the_loss = loss + self.args.reg_coef * reg_cost
                self.optim.zero_grad()
                the_loss.backward()
                self.optim.step()
            loss_ = loss_ + loss.item()
        return loss_

    def Train_model(self):
        epoch = self.args.epoch
        test_RMSE = []
        test_score = []
        RUL_predicted = []
        RUL_real = []
        earlyExit_rmse = 1e10
        earlyExit_cur = 0
        for i in range(epoch):
            loss = self.Train_batch()
            self.lr_schedular.step(i)
            if i % self.args.show_interval == 0:
                print(self.args.save_name, flush=True)
                print('The the {}th epoch, Training Loss is {}'.format(i, loss), flush=True)
                print('Learning Rate is', self.lr_schedular.get_lr(), flush=True)
                test_RMSE_, test_score_, test_result_predicted, test_result_real = self.Prediction()
                print('TESTING RMSE is {}, TESTING Score is {}'.format(test_RMSE_, test_score_), flush=True)
                test_RMSE.append(test_RMSE_)
                test_score.append(test_score_)
                RUL_predicted.append(test_result_predicted)
                RUL_real.append(test_result_real)
                if test_RMSE_ < earlyExit_rmse:
                    earlyExit_rmse = test_RMSE_
                    earlyExit_cur = i
                elif i - earlyExit_cur > 10:
                    print("Early exit ... \n", flush=True)
                    break
        best_score = min(test_score)
        best_score_rmse = test_RMSE[test_score.index(best_score)]
        return int(best_score), best_score_rmse

    def cuda_(self, x):
        x = tr.Tensor(np.array(x))

        if tr.cuda.is_available():
            return x.to(tr.device(self.args.device))
        else:
            return x

    def Prediction(self):
        self.net.eval()
        predicted_rul = []
        for data, label, bs_corrcoef, heter_types in self.test_loader:
            prediction, _ = self.net(data, bs_corrcoef, heter_types)
            predicted_RUL = tr.squeeze(prediction)
            predicted_rul.append(predicted_RUL.detach().cpu())
            # print(predicted_RUL)

        real_RUL = self.test_label.cpu()
        predicted_RUL = tr.cat(predicted_rul, 0)
        MSE = self.loss_function(predicted_RUL, real_RUL)
        RMSE = tr.sqrt(MSE) * self.max_RUL
        RMSE = RMSE.detach().cpu().numpy()

        score = self.scoring_function(predicted_RUL, real_RUL)
        # score = score.detach().cpu().numpy()
        score = score.item()

        return RMSE, score, predicted_RUL, real_RUL

    def visualization(self, prediction, real):
        fig = plt.figure()
        sub = fig.add_subplot(1, 1, 1)

        sub.plot(prediction, color='red', label='Predicted Labels')
        sub.plot(real, 'black', label='Real Labels')
        sub.legend()
        plt.show()

    def scoring_function(self, predicted, real):
        score = 0
        num = predicted.size(0)
        for i in range(num):
            if real[i] > predicted[i]:
                score = score + (tr.exp((real[i] * self.args.max_rul - predicted[i] * self.args.max_rul) / 13) - 1)

            elif real[i] <= predicted[i]:
                score = score + (tr.exp((predicted[i] * self.args.max_rul - real[i] * self.args.max_rul) / 10) - 1)

        return score


def train_CMPS(args):
    # for HTGNN
    args.data_name = 'CMPS'
    args.data_path = './CMAPSS/'
    args.epoch = 51
    args.max_rul = 125
    args.batch_size = 50 #10
    args.num_nodes = 14
    args.feature_dimension = 1
    args.time_length = 50
    # args.hid_dim = 32
    # args.cor_embed_dim = 16
    # args.num_rnn_layers = 2
    args.num_rnn_layers = 1
    # args.lr = 0.005
    args.num_node_type = 8

    # In CMAPSS, there 4 sub-dataset indicated by test_idx
    for test_idx in range(1, 5):
        args.data_sub = test_idx
        rmse_lst = []
        score_lst = []
        # Repeat 5 experiments
        # for i in range(5):
        for i in range(5):
            args.save_name = 'HTGNN_{}_{}_{}'.format(args.data_name, test_idx, i)
            print(args, flush=True)
            setup_seed(seed_lst[i])
            train = Train(args)
            score, rmse = train.Train_model()
            print('The {} th experiment: Score={}, RMSE={}'.format(str(i), str(score), str(rmse)), flush=True)
            rmse_lst.append(rmse)
            score_lst.append(score)
        # Results log
        # with open(r'./experiment' + \
        #           '/{}-{}-{}.txt'.format(args.model_name, args.data_name, args.experiName), 'a') as fp:
        #     res_str = 'Dataset id: ' + str(test_idx) + '\n' + \
        #               'avg RMSE: ' + str(round(sum(rmse_lst) / len(rmse_lst), 2)) + '\n' + \
        #               'avg Score: ' + str(int(sum(score_lst) / len(score_lst))) + '\n' + \
        #               'max RMSE: ' + str(round(float(max(rmse_lst)), 2)) + ' ' + \
        #               'max Score: ' + str(int(max(score_lst))) + '\n' + \
        #               'min RMSE: ' + str(round(float(min(rmse_lst)), 2)) + ' ' + \
        #               'min Score: ' + str(int(min(score_lst))) + '\n \n'
        #     fp.write(res_str)
        #     print(res_str, flush=True)

        sta_list.append(
            [str(test_idx), str(round((sum(rmse_lst)-max(rmse_lst)-min(rmse_lst)) / (len(rmse_lst)-2), 2)), str(int((sum(score_lst)-max(score_lst)-min(score_lst)) / (len(score_lst)-2))),
             str(round(float(max(rmse_lst)), 2)), str(int(max(score_lst))), str(round(float(min(rmse_lst)), 2)),
             str(int(min(score_lst)))])


def train_NCMPS(args):
    args.data_name = 'NCMPS'
    args.data_path = './N-CMAPSS/Samples_whole'
    args.epoch = 51
    # args.epoch = 1
    args.batch_size = 50
    args.num_nodes = 20
    args.feature_dimension = 1
    args.max_RUL = 88
    args.time_length = 50
    # args.hid_dim = 32
    # args.cor_embed_dim = 16
    args.num_rnn_layers = 1
    args.num_node_type = 7
    # args.typeFunc_dim = 16
    # args.lr = 0.001

    for test_idx in range(4):
        rmse_lst = []
        score_lst = []
        args.data_sub = test_idx

        # for i in range(5):
        for i in range(5):
            args.save_name = 'HTGNN_{}_{}_{}'.format(args.data_name, test_idx, i)
            print(args, flush=True)
            setup_seed(seed_lst[i])
            train = Train(args)
            score, rmse = train.Train_model()
            print('The {} th experiment: Score={}, RMSE={} \n'.format(str(i), str(score), str(rmse)), flush=True)
            rmse_lst.append(rmse)
            score_lst.append(score)
        with open(r'./experiment' + \
                  '/{}-{}-{}.txt'.format(args.model_name, args.data_name, args.experiName), 'a') as fp:
            res_str = 'Dataset id: ' + str(test_idx) + '\n' + \
                      'avg RMSE: ' + str(round(sum(rmse_lst) / len(rmse_lst), 2)) + '\n' + \
                      'avg Score: ' + str(int(sum(score_lst) / len(score_lst))) + '\n' + \
                      'max RMSE: ' + str(round(float(max(rmse_lst)), 2)) + ' ' + \
                      'max Score: ' + str(int(max(score_lst))) + '\n' + \
                      'min RMSE: ' + str(round(float(min(rmse_lst)), 2)) + ' ' + \
                      'min Score: ' + str(int(min(score_lst))) + '\n \n'
            fp.write(res_str)
            print(res_str, flush=True)

        sta_list.append(
            [str(test_idx), str(round((sum(rmse_lst) - max(rmse_lst) - min(rmse_lst)) / (len(rmse_lst) - 2), 2)),
             str(int((sum(score_lst) - max(score_lst) - min(score_lst)) / (len(score_lst) - 2))),
             str(round(float(max(rmse_lst)), 2)), str(int(max(score_lst))), str(round(float(min(rmse_lst)), 2)),
             str(int(min(score_lst)))])


if __name__ == '__main__':
    from args import args

    args = args()
    start = time.perf_counter()
    sta_list = []
    if args.data_name == 'CMPS':
        train_CMPS(args)
    elif args.data_name == 'NCMPS':
        train_NCMPS(args)
    end = time.perf_counter()
    print("\n time consuming {:.2f}".format(end - start))

    with open('./experiment' + '/{}-{}-coef={}-{}-{}.csv'.format(args.data_name, args.typeFunc_dim, args.reg_coef, args.lr, args.experiName), 'w') as f:
        f.write('Dataset id: ')
        f.write(',')
        f.write('avg RMSE: ')
        f.write(',')
        f.write('avg Score: ')
        f.write(',')
        f.write('max RMSE: ')
        f.write(',')
        f.write('max Score: ')
        f.write(',')
        f.write('min RMSE: ')
        f.write(',')
        f.write('min Score: ')
        f.write('\n')

        for a in sta_list:
            for b in range(len(a)):
                if b != len(a) - 1:
                    f.write(a[b])
                    f.write(',')
                else:
                    f.write(a[b])
                    f.write('\n')





