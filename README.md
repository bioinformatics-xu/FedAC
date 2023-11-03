# FedAC
This is the code of paper [FedAC:Improving Acute Coronary Syndrome Prediction through Federated Adaptive Clustering].

## Usage
An example to run FedAC:
```
python experiments.py --model=simple-cnn \
    --dataset=generated \
    --alg=FedAC \
    --lr=0.005 \
    --batch-size=32 \
    --epochs=50 \
    --n_parties=5 \
    --mu=0.01 \
    --rho=0.9 \
    --comm_round=50 \
    --partition=noniid-labeldir \
    --beta=0.5\
    --device='cpu'\
    --datadir='./data/' \
    --logdir='./logs/' \
    --noise=0 \
    --sample=1 \
    --init_seed=0
```

## Description of parameters
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. Options: `simple-cnn`, `vgg`, `resnet`, `mlp`. Default = `mlp`. |
| `dataset`      | Dataset to use. Options: `generated`. Default = `generated`. |
| `alg` | The training algorithm. Default = `FedAC`. |
| `lr` | Learning rate for the local models, default = `0.005`. |
| `batch-size` | Batch size, default = `32`. |
| `epochs` | Number of local training epochs, default = `50`. |
| `n_parties` | Number of parties, default = `5`. |
| `mu` | The proximal term parameter for FedProx, default = `1`. |
| `rho` | The parameter controlling the momentum SGD, default = `0`. |
| `comm_round`    | Number of communication rounds to use, default = `50`. |
| `partition`    | The partition way. Options:  `noniid-labeldir`. Default = `noniid-labeldir` |
| `beta` | The concentration parameter of the Dirichlet distribution for heterogeneous partition, default = `0.5`. |
| `device` | Specify the device to run the program, default = `cuda:0`. |
| `datadir` | The path of the dataset, default = `./data/`. |
| `logdir` | The path to store the logs, default = `./logs/`. |
| `noise` | Maximum variance of Gaussian noise we add to local party, default = `0`. |
| `sample` | Ratio of parties that participate in each communication round, default = `1`. |
| `init_seed` | The initial seed, default = `0`. |

## Cite
This project is developed based on Non-NllD, if you find this repository useful, please cite paper:
```
@inproceedings{li2022federated,
      title={Federated Learning on Non-IID Data Silos: An Experimental Study},
      author={Li, Qinbin and Diao, Yiqun and Chen, Quan and He, Bingsheng},
      booktitle={IEEE International Conference on Data Engineering},
      year={2022}
}
```

## Developor

Liang Xu (2636802625@qq.com)

Xiaolu Xu (lu.xu@lnnu.edu.cn)

School of Computer and Artificial Intelligence 

Liaoning Normal University







