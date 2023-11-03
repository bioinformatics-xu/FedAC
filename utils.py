import os
import logging
import numpy as np
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, f1_score

from model import *
from datasets import Generated, sick_binary

import random
from sklearn import metrics

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts

def partition_data(dataset, datadir, logdir, partition, n_parties, beta):
    if datadir == "./data/sick_binary":
        print("")
        cell_features = pd.read_csv(datadir + '/sick_binary_feature.csv')
        x_origin = cell_features.iloc[:, :]
        scaler = preprocessing.MinMaxScaler()
        x_features = scaler.fit_transform(x_origin)
        x_origin = pd.DataFrame(x_features, columns=x_origin.columns)
        x_origin = x_origin[x_origin.columns.sort_values()]
        cardio_class = pd.read_csv(datadir + '/sick_binary_class.csv')
        y_origin = cardio_class['cardio']

        X_train, X_test, y_train, y_test = train_test_split(x_origin, y_origin, test_size=0.3, random_state=66)
        X_train = np.array(X_train, dtype='float32')
        X_test = np.array(X_test, dtype='float32')
        y_train = np.array(y_train, dtype='int64')
        y_test = np.array(y_test, dtype='int64')

        if os.path.exists(datadir):
            pass
        else:
            mkdirs(datadir)

        np.save(datadir + "/X_train.npy", X_train)
        np.save(datadir + "/X_test.npy", X_test)
        np.save(datadir + "/y_train.npy", y_train)
        np.save(datadir + "/y_test.npy", y_test)

    elif dataset == 'generated':
        X_train, y_train = [], []
        for loc in range(4):
            for i in range(1000):
                p1 = random.random()
                p2 = random.random()
                p3 = random.random()
                if loc > 1:
                    p2 = -p2
                if loc % 2 ==1:
                    p3 = -p3
                if i % 2 == 0:
                    X_train.append([p1, p2, p3])
                    y_train.append(0)
                else:
                    X_train.append([-p1, -p2, -p3])
                    y_train.append(1)
        X_test, y_test = [], []
        for i in range(1000):
            p1 = random.random() * 2 - 1
            p2 = random.random() * 2 - 1
            p3 = random.random() * 2 - 1
            X_test.append([p1, p2, p3])
            if p1>0:
                y_test.append(0)
            else:
                y_test.append(1)
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        y_test = np.array(y_test, dtype=np.int64)
        idxs = np.linspace(0,3999,4000,dtype=np.int64)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        mkdirs("data/generated/")
        np.save("data/generated/X_train.npy",X_train)
        np.save("data/generated/X_test.npy",X_test)
        np.save("data/generated/y_train.npy",y_train)
        np.save("data/generated/y_test.npy",y_test)
    n_train = y_train.shape[0]

    if partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY', 'heart' ):
            K = 2
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200

        N = y_train.shape[0]
        N_test = y_test.shape[0]
        net_dataidx_map = {}
        net_dataidx_map_test = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            idx_batch_test = [[] for _ in range(n_parties)]
            idx_k_0 = np.where(y_train == 0)[0]
            np.random.shuffle(idx_k_0)
            idx_k_test_0 = np.where(y_test == 0)[0]
            np.random.shuffle(idx_k_test_0)
            idx_k_1 = np.where(y_train == 1)[0]
            np.random.shuffle(idx_k_1)
            idx_k_test_1 = np.where(y_test == 1)[0]
            np.random.shuffle(idx_k_test_1)
            if n_parties == 5:
                proportions_0 = [0.25,0.15,0.16,0.17,0.27]
                proportions_test_0 = proportions_0
                proportions_1 = [0.25,0.15,0.16,0.17,0.27]
                proportions_test_1 = proportions_1
                # proportions_0 = [0.18 0.21, 0.19, 0.2, 0.22]
                # proportions_test_0 = proportions_0
                # proportions_1 = [0.14, 0.245, 0.215, 0.23, 0.17]
                # proportions_test_1 = proportions_1
            proportions_0 = (np.cumsum(proportions_0) * len(idx_k_0)).astype(int)[:-1]
            proportions_test_0 = (np.cumsum(proportions_test_0) * len(idx_k_test_0)).astype(int)[:-1]
            proportions_1 = (np.cumsum(proportions_1) * len(idx_k_1)).astype(int)[:-1]
            proportions_test_1 = (np.cumsum(proportions_test_1) * len(idx_k_test_1)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k_0, proportions_0))]
            idx_batch_test = [idx_j_test + idx_test.tolist() for idx_j_test, idx_test in zip(idx_batch_test, np.split(idx_k_test_0, proportions_test_0))]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k_1, proportions_1))]
            idx_batch_test = [idx_j_test + idx_test.tolist() for idx_j_test, idx_test in zip(idx_batch_test, np.split(idx_k_test_1, proportions_test_1))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            min_size_test = min([len(idx_j_test) for idx_j_test in idx_batch_test])
        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            np.random.shuffle(idx_batch_test[j])
            net_dataidx_map_test[j] = idx_batch_test[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test, logdir)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts)

def compute_accuracy(model, dataloader, get_confusion_matrix=False, moon_model=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total, total_0_0, total_0_1, total_1_0, total_1_1 = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                if moon_model:
                    _, _, out = model(x)
                else:
                    out = model(x)
                pred_num, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_nums_list = np.append(pred_labels_list, pred_num.numpy())
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_nums_list = np.append(pred_labels_list, pred_num.cpu().numpy())
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    auc = metrics.roc_auc_score(true_labels_list, pred_nums_list)
    f1 = f1_score(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), auc, f1, conf_matrix

    return correct/float(total), auc, f1

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, dataidxs_test=None, noise_level=0, net_id=None, total=0):
    if dataset == 'generated':
        dl_obj = Generated
        transform_train = None
        transform_test = None
    elif dataset == 'sick_binary':
        dl_obj = sick_binary
        transform_train = None
        transform_test = None
    if dataset in ('mnist', 'femnist', 'fmnist', 'svhn', 'cifar10', 'cifar100', 'generated'):
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)
    else:
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)
    return train_dl, test_dl, train_ds, test_ds