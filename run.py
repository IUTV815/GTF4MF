import argparse
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
import os
import psutil
def use_cpus(gpus: list, cpus_per_gpu: int):
        cpus = []
        for gpu in gpus:
            cpus.extend(list(range(gpu* cpus_per_gpu, (gpu+1)* cpus_per_gpu)))
        p = psutil.Process()
        p.cpu_affinity(cpus)
        print("A total {} CPUs are used, making sure that num_worker is small than the number of CPUs".format(len(cpus)))
      

def seed_torch(seed=2025):
    random.seed(seed)   
    np.random.seed(seed)   
    torch.manual_seed(seed)   

def train(ii,args):
    setting = '{}_{}_{}_seq{}_pred{}'.format(
        args.data_type,
        args.task,
        args.model,
        args.seq_len,
        args.pred_len,
        )

    exp = Exp(args)  # set experiments
    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    exp.train(setting)
    print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
    exp.test(setting,test=1)
    print(f'>>>>>>>visualizing of forecasting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
    exp.visual_forecasting(setting,test=1)
    
if __name__ == '__main__':
    seed_torch(2025)
    parser = argparse.ArgumentParser(description='Leddam')

    # basic config
    parser.add_argument('--model', type=str, required=True, default='Leddam')
    # data loader
    parser.add_argument('--data_type', type=str, required=False, default='Male', help='dataset type')
    parser.add_argument('--data', type=str, required=True, default='Mortality', help='dataset')
    parser.add_argument('--data_path', type=str, default='./dataset/log_log_Female_Mortality.csv', help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--task', type=str, default='MM', 
                        help='forecasting tasks, choose from [MM: (Multivariate 2 Multivariate) SS: (Univariate 2 Univariate)]')
    parser.add_argument('--seq_len', type=int, default=36, help='input sequence length of backbone model')
    parser.add_argument('--pred_len', type=int, default=2, help='prediction sequence length')
    parser.add_argument('--d_model', type=int, default=512, help='prediction sequence length')
    parser.add_argument('--d_ff', type=int, default=512, help='prediction sequence length')
    parser.add_argument('--layers', type=int, default=2, help='prediction sequence length')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='feedfoward dropout ratio')
    
    # model define
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm (RevIN)')
    parser.add_argument('--enc_in', type=int, default=24, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=24, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=24, help='output size')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument("--gpu_idx", nargs="+", type=int, default=[0,1,2,3,4,5,6,7], help="List of GPU indices to use")
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')


    args = parser.parse_args()
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    args.label_len=int(args.seq_len//2)
    args.itr=1

    if args.task == 'SS':
        args.enc_in=1
        args.c_out=1
    if args.data_path[-18:-14]!='Male':
        args.data_type='Female'
    
    # For server 1, set cpus_per_gpu to 12
    #For server 2, set cpus_per_gpu to 24
    use_cpus(gpus=args.gpu_idx, cpus_per_gpu=12)
    
    print('Args in experiment:')
    print(args)
    Exp = Exp_Long_Term_Forecast
    train(1,args)
