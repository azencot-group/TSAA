from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Custom_aa, Dataset_ETT_minute_aa
from torch.utils.data import DataLoader
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'custom_aa':Dataset_Custom_aa,
    'ETTm2_aa': Dataset_ETT_minute_aa,
}


def data_provider(args, flag, generator = None, kfold = 1, kfold_i = 1, kfold_flag = 'train'):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.detail_freq
        Data = Dataset_Pred
        
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if flag == 'train':    
        stl = args.stl
    else:
        stl = False

    seasonality = args.seasonality
    id = args.task_id.split('_')
    dataset = id[0] 
    horzion = id[-1] 
    stl_name = f'{dataset}_{horzion}_{args.model}'

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        stl=stl,
        stl_name=stl_name,
        seasonality=seasonality,
        kfold = kfold,
        kfold_i = kfold_i,
        kfold_flag = kfold_flag,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        generator = generator)
    return data_set, data_loader
