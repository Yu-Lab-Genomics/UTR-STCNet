#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1" #  4, 5, 6

from UTR_MT.utr_dataset_all import *
from UTR_MT.dataloader_mt import *
from UTR_MT.UTRFormer_MT import utrformer_large as utrformer
from copy import deepcopy
from tqdm import tqdm, trange
import torch
from torchsummary import summary
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
from itertools import cycle
from util import *
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import  spearmanr
from torch.optim.lr_scheduler import CyclicLR
random.seed(1337)


parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', type=str, default='0', help="Training Devices")
parser.add_argument('--local_rank', type=int, default=0, help="DDP parameter, do not modify")

parser.add_argument('--prefix', type=str, default = 'UTR_MAP')  
parser.add_argument('--label-class', type=str, default = 'rl')  
parser.add_argument('--seq_type', type=str, default = 'utr') 
parser.add_argument('--epochs', type = int, default = 300)
parser.add_argument('--layers', type = int, default = 16)  
parser.add_argument('--heads', type = int, default = 16)
parser.add_argument('--embed_dim', type = int, default = 128)
parser.add_argument('--batchsize', type = int, default = 256)
parser.add_argument('--seq_maxlen', type = int, default = 120)
parser.add_argument('--batch-tokens', type = int, default = 12800)
parser.add_argument('--supervised_weight', type = float, default = 1)
parser.add_argument('--structure_weight', type = float, default = 1)
parser.add_argument('--train_setting', type = str, default = 'mt.yaml') 
parser.add_argument('--task', type = str, required = True) 
parser.add_argument('--test_id', type = int, required = True)
parser.add_argument('--num_datasets', type = int, required = True)
parser.add_argument('--task_weights',type=float,nargs='+',required=True,  help='task weights for each dataset')
parser.add_argument('--lr', type = float, default = 1e-5)
parser.add_argument('--load_wholemodel', action = 'store_true') ## if --: True
parser.add_argument('--modelfile', type = str, default = '')
parser.add_argument('--init_epochs', type = int, default = 0)

args = parser.parse_args()

global idx_to_tok, prefix, epochs, layers, heads, embed_dim, batch_toks, device, repr_layers, evaluation, include, truncate, return_contacts, return_representation, mask_toks_id ,device_ids, structure_tok_to_idx
# print("*"*20, args.train_fasta)

device_ids = list(map(int, args.device_ids.split(',')))
device = torch.device('cuda:{}'.format(device_ids[args.local_rank]))
torch.cuda.set_device(device)


prefix = f'{args.prefix}_{args.label_class.upper()}'
epochs = args.epochs
# train_fasta = args.train_fasta

outputfilename = f'{prefix}_{args.task}_epoch{args.epochs}_batchsize{args.batchsize}_lr{args.lr}'
print(outputfilename)


file = load_yaml(args.train_setting)[args.task]
task_num = len(file["train_file"])
print(args.seq_maxlen)
concat_dataset = ConcatDataset([DatasetWrapper(f"{item}", idx, args.seq_maxlen) for idx, item in enumerate(file["train_file"])])
dataset_sizes = [len(item) for item in concat_dataset.datasets]

sampler = DatasetSampler(
    dataset_sizes=dataset_sizes,
    main_id=file["main_task"],                       
    main_samples=args.batchsize,                     
    other_samples=args.batchsize*0.1,               
)
train_batches_loader = DataLoader(
    concat_dataset,
    batch_sampler=sampler,
    num_workers=8,
    pin_memory=True
)

val_dataset = ConcatDataset([DatasetWrapper(f"{item}", idx, args.seq_maxlen) for idx, item in enumerate(file["val_file"])])
batches_loader2 = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size = args.batchsize, # args.batchsize; 1
                                             num_workers = 8,
                                             shuffle = False
                                             )

model = utrformer(task_num = task_num).to(device)
if args.load_wholemodel: 
    model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(args.modelfile)["model"].items()}) # , strict=False


if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

if len(args.task_weights) != args.num_datasets:
    raise ValueError(
        f"Expected {args.num_datasets} task weights, but got {len(args.task_weights)}"
    )
task_weights = args.task_weights 


optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=args.lr,
                            betas=(0.9, 0.98), 
                            eps=1e-09, 
                            weight_decay=1e-4)

scheduler=CyclicLR(optimizer, base_lr=args.lr, max_lr=args.lr*10, step_size_up=10, step_size_down=50)

loss_supervised_best, ep_best = np.inf,  -1
loss_supervised_list = []

criterions = nn.ModuleList([nn.MSELoss().to(device) for _ in range(args.num_datasets)])

dir_saver = f'./checkpoint/{args.prefix}/'
if not os.path.exists(dir_saver):
    os.makedirs(dir_saver)

if not os.path.exists('./figures/'): os.makedirs('./figures/')
model.train()

r2_best, spr_best, rmse_best = float('-inf'), float('-inf'), float('inf')


for epoch in range(args.init_epochs+1, args.init_epochs + epochs + 1):
    loss_supervised_epoch = []
    with tqdm(total=len(train_batches_loader)) as t:
        for strs, tokens, labels, dataset_id  in train_batches_loader:
            t.set_description(f'Epoch {epoch} ')
            input = tokens.to(device)
            targets = torch.tensor(labels, dtype=torch.float).to(device)
            optimizer.zero_grad()

            out, out_index = model(input, dataset_id)
            loss = 0
            for i in range(args.num_datasets):
                if i < len(out_index):
                    loss += task_weights[i] * criterions[i](out[i].squeeze(), targets[out_index[i]])

            loss_supervised_epoch.append(loss.cpu().detach().tolist())
            loss.backward()
            optimizer.step()
            scheduler.step()

            t.set_postfix(loss=loss.item(), lr=round(optimizer.param_groups[0]['lr'], 5))
            t.update()
    
    if epoch % 1 == 0:
        true_label = []
        predict_label = []
        with torch.no_grad():
            for _, batch in enumerate(tqdm(batches_loader2)):
                strs, tokens, labels, dataset_id = batch
                input = tokens.to(device)
                targets = torch.tensor(labels).to(device)
                out, _ = model(input, dataset_id, train=False,test_id=args.test_id)
                true_label += targets.tolist()
                if len(out)==1:
                    predict_label.append(out.item())
                else:
                    predict_label +=  out.squeeze().cpu().detach().tolist()
        y_true = true_label
        y_pred = predict_label
        r2 = r2_score(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f">>>>> val-: val dataset R-squared: {round(r2, 4)}   Spearman R: {round(spearman_corr, 4)}   RMSE: {round(rmse, 4)}  LR: {round(optimizer.param_groups[0]['lr'], 7)}\n") 
    # if epoch == 20:
    #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1

    loss_supervised_epoch = np.mean(loss_supervised_epoch)
    loss_supervised_list.append(loss_supervised_epoch)
    
    print(f'>>>>> Epoch = {epoch}: Loss = {loss_supervised_epoch:.4f}')


    if r2>r2_best: 
        r2_best = r2
        loss_supervised_best, ep_best = loss_supervised_epoch, epoch
        path_saver = dir_saver + f'{outputfilename}.pkl'
        print(f'****Saving model in {path_saver}: \nBest epoch = {ep_best}')
        ckpt = {
            "model": model.state_dict(),
        }
        torch.save(ckpt, path_saver)
        if epoch > 20 and optimizer.param_groups[0]['lr']>1e-5:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1
    
    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 10))
    axes.plot(range(epoch-args.init_epochs), loss_supervised_list, label = 'Supervised Info|MSELoss')
    plt.title(f'{outputfilename}')
    plt.legend()
    plt.savefig(f'./figures/{outputfilename}.tif')

