import json
import torch.optim as optim
import argparse
from math import *
import random
from sklearn.cluster import KMeans
import datetime
from utils import *
from resnetcifar import *

#轮廓系数
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
def lun(X):
    n_clusters_range = range(2,5)
    silhouette_scores = []
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
    ic = 0
    for i in range(3):
        if silhouette_scores[ic] < silhouette_scores[i]:
            ic = i
        print(silhouette_scores[i])
    ic = ic+2
    return ic

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='generated', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='alg')
    parser.add_argument('--batch-size', type=int, default=30, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=50, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=5,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='FedAC',
                            help='fl algorithms: FedAC')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=60, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedac')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cpu', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0.9, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    args = parser.parse_args()
    return args

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}
    if args.alg == 'moon':
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon_noheader(args.model+add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        for net_i in range(n_parties):
            if args.dataset == "generated":
                net = PerceptronModel()
            elif args.model == "mlp":
                if args.dataset == './data/sick_binary':
                    input_size = 54
                    output_size = 2
                    hidden_sizes = [200,100,50]
                net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
            else:
                print("not supported yet")
                exit(1)
            nets[net_i] = net
    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type

#fedavg全局模型

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc, train_auc, train_f1 = compute_accuracy(net, train_dataloader, device=device)
    test_acc, test_auc, test_f1, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))
    logger.info('>> Pre-Training Training AUC: {}'.format(train_auc))
    logger.info('>> Pre-Training Test AUC: {}'.format(test_auc))
    logger.info('>> Pre-Training Training f1: {}'.format(train_f1))
    logger.info('>> Pre-Training Test f1: {}'.format(test_f1))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)
    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]


    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc, train_auc, train_f1 = compute_accuracy(net, train_dataloader, device=device)
    test_acc, test_auc, test_f1, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    print(conf_matrix)


    logger.info(' ** Training complete **')
    return train_acc, test_acc, train_auc, test_auc, train_f1, test_f1

#fedac本地训练模型
def local_train_net(nets, selected, args, net_dataidx_map,net_dataidx_map_test, test_dl = None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]
        dataidxs_test = net_dataidx_map_test[net_id]
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(net_dataidx_map[net_id])))
        logger.info("Testing network %s. n_testing: %d" % (str(net_id), len(net_dataidx_map_test[net_id])))
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, dataidxs_test, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, dataidxs_test, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        trainacc, testacc, trainauc, testauc, trainf1, testf1 = train_net(net_id, net, train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        logger.info("net %d final test auc %f" % (net_id, testauc))
        logger.info("net %d final test f1 %f" % (net_id, testf1))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

#训练数据集
def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map

if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts = partition_data(args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,args.datadir,args.batch_size,32)
    train_dl_global_2 = train_dl_global_1 = train_dl_global_0 = train_dl_global
    test_dl_global_2 = test_dl_global_1 = test_dl_global_0 = test_dl_global

    print("len train_dl_global:", len(train_ds_global))
    print("len test_dl_global:", len(test_ds_global))


    data_size = len(test_ds_global)

    result_acc_in = []
    result_acc = []
    result_auc_in = []
    result_auc = []
    result_f1_in = []
    result_f1 = []
    result_acc_avg = []
    result_auc_avg = []
    result_f1_avg = []

    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)
    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
    global_para_0 = {}
    global_para_1 = {}
    global_para_2 = {}
    global_para_3 = {}
    global_para_4 = {}
    global_para_5 = {}
    global_para_avg = {}
    global_para_old = {}
    global_model = global_models[0]
    global_para = global_model.state_dict()
    for key in global_para:
        global_para_avg[key] = global_para[key]
    for key in global_para:
        global_para_old[key] = global_para[key]
    train_global_in_list = []
    test_global_in_list = []

    if args.alg == 'FedAC':
        global_para_all = {}
        global_model_all = {}
        for i in range(args.n_parties):
            global_para_all[i] = global_para
            global_model_all[i] = global_model

        label = np.zeros([args.n_parties])
        ks = 1
        ks_all = []
        label_all = []

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            all_selected = arr[:int(args.n_parties * args.sample)]
            selected = []
            # 1.1
            for q in range(ks):
                selected.clear()
                for w in range(args.n_parties):
                    if label[w] == q:
                        selected.append(w)
                if round == 0:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para_old)
                else:
                    if q == 0:
                        for key in global_para:
                            global_para[key] = global_para_0[key]
                    elif q == 1:
                        for key in global_para:
                            global_para[key] = global_para_1[key]
                    elif q == 2:
                        for key in global_para:
                            global_para[key] = global_para_2[key]
                    elif q == 3:
                        for key in global_para:
                            global_para[key] = global_para_3[key]
                    elif q == 4:
                        for key in global_para:
                            global_para[key] = global_para_4[key]
                    elif q == 5:
                        for key in global_para:
                            global_para[key] = global_para_5[key]
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
                local_train_net(nets, selected, args, net_dataidx_map, net_dataidx_map_test, test_dl=test_dl_global,device=device)
                net_paramter = []
            for idx in range(args.n_parties):
                net_para = nets[all_selected[idx]].cpu().state_dict()
                net_param = np.array([])
                for key in net_para:
                    net_param = np.append(net_param, net_para[key])
                net_paramter += [net_param]
            net_paramter_matrix = np.array(net_paramter)
            ks = lun(net_paramter_matrix)
            ks_all.append(ks)
            print(ks)
            julei = KMeans(n_clusters=ks, random_state=1)
            julei.fit(net_paramter)
            label = julei.labels_

            print(label)
            label_all.append(label)
            print(label_all)
            for e in range(ks):
                selected.clear()
                for r in range(args.n_parties):
                    if label[r] == e:
                        selected.append(r)
                total_data_points = sum([len(net_dataidx_map[s]) for s in selected])
                fed_avg_freqs = [len(net_dataidx_map[s]) / total_data_points for s in selected]

                for key in global_para:
                    global_para[key] = global_para[key] * 0
                for i in selected:
                    net_para = nets[i].cpu().state_dict()
                    count = 0
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[count]
                    count += 1
                if e == 0:
                    for key in global_para:
                        global_para_0[key] = global_para[key]
                elif e == 1:
                    for key in global_para:
                        global_para_1[key] = global_para[key]
                elif e == 2:
                    for key in global_para:
                        global_para_2[key] = global_para[key]
                elif e == 3:
                    for key in global_para:
                        global_para_3[key] = global_para[key]
                elif e == 4:
                    for key in global_para:
                        global_para_4[key] = global_para[key]
                elif e == 5:
                    for key in global_para:
                        global_para_5[key] = global_para[key]
                global_model.load_state_dict(global_para)
                global_model.to(device)

                train_global_in_list.clear()
                test_global_in_list.clear()
                for label_i in selected:
                    center = julei
                    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset, args.datadir,args.batch_size,32,net_dataidx_map[label_i],net_dataidx_map_test[label_i])
                    train_global_in_list.append(train_ds_global)
                    test_global_in_list.append(test_ds_global)
                train_global_in_ds = data.ConcatDataset(train_global_in_list)
                train_dl_global = data.DataLoader(dataset=train_global_in_ds, batch_size=args.batch_size, shuffle=True)
                test_global_in_ds = data.ConcatDataset(test_global_in_list)
                test_dl_global = data.DataLoader(dataset=test_global_in_ds, batch_size=args.batch_size, shuffle=True)

                logger.info('global n_training: %d' % len(train_dl_global))
                logger.info('global n_test: %d' % len(test_dl_global))
                train_acc, train_auc, train_f1 = compute_accuracy(global_model, train_dl_global, device=device)
                test_acc, test_auc, test_f1, conf_matrix = compute_accuracy(global_model, test_dl_global,get_confusion_matrix=True, device=device)
                logger.info('>> Global Model %d Train accuracy: %f' % (e, train_acc))
                logger.info('>> Global Model %d Test  accuracy: %f' % (e, test_acc))
                logger.info('>> Global Model %d Train AUC: %f' % (e, train_auc))
                logger.info('>> Global Model %d Test  AUC: %f' % (e, test_auc))
                logger.info('>> Global Model %d Train f1: %f' % (e, train_f1))
                logger.info('>> Global Model %d Test  f1: %f' % (e, test_f1))
                print(conf_matrix)
                result_acc_in.append(test_acc)
                result_auc_in.append(test_auc)
                result_f1_in.append(test_f1)

            result_acc_in_arr = np.array(result_acc_in)
            result_acc_in.clear()
            result_auc_in_arr = np.array(result_auc_in)
            result_auc_in.clear()
            result_f1_in_arr = np.array(result_f1_in)
            result_f1_in.clear()
            result_acc.append(result_acc_in_arr)
            result_auc.append(result_auc_in_arr)
            result_f1.append(result_f1_in_arr)
        print(result_acc)