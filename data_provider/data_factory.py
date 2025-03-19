from data_provider.data_loader import Dataset_Mortality
from torch.utils.data import DataLoader
import torch
import numpy as np

data_dict = {
    'Mortality': Dataset_Mortality,
}


def data_provider(flag,args):
    Data = data_dict[args.data]
    drop_last=False
    
    if flag == 'test':
        drop_last=False
        shuffle_flag = False
        batch_size = args.batch_size   
    else:
        drop_last=True
        shuffle_flag = True
        batch_size = args.batch_size  

    dataset = Data(flag=flag, seq_len=args.seq_len, pred_len=args.pred_len,data_path=args.data_path)
    print(flag, len(dataset))
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        )
    return data_loader