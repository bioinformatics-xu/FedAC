import os
import logging
import numpy as np
import random
import argparse
import csv

from utils import mkdirs
def partition_data(dataset, class_id, K, partition, n_parties, beta, seed):
    np.random.seed(seed)
    random.seed(seed)

    n_train = dataset.shape[0]
    y_train = dataset[:,class_id]
    if partition == "noniid-labeldir":
        min_size = 0
        min_require_size = 10

        N = dataset.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # logger.info("proportions1: ", proportions)
                # logger.info("sum pro1:", np.sum(proportions))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                # logger.info("proportions2: ", proportions)
                proportions = proportions / proportions.sum()
                # logger.info("proportions3: ", proportions)
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # logger.info("proportions4: ", proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break


        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--n_parties', type=int, default=10,  help='number of workers in a distributed cluster')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--datadir', type=str, required=False, default="./data/creditcard.csv", help="Data directory")
    parser.add_argument('--outputdir', type=str, required=False, default="./data/creditcard/", help="Output directory")
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    num = -1
    dataset = []
    with open(args.datadir, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if num == -1:
                header = row
            else:
                dataset.append(row)
                for i in range(len(dataset[-1])):
                    dataset[-1][i] = eval(dataset[-1][i])
            num += 1
    
    class_id = 0
    for i in range(len(header)):
        if header[i] == "Class":
            class_id = i
            break
    dataset = np.array(dataset)
    num_class = int(np.max(dataset[:,class_id])) + 1

    net_dataidx_map = partition_data(dataset, class_id, num_class, args.partition, args.n_parties, args.beta, args.init_seed)
    mkdirs(args.outputdir)
    for i in range(args.n_parties):
        file_name = args.outputdir+str(i)+'.csv'
        os.system("touch "+file_name)
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(dataset[net_dataidx_map[i]])

            
