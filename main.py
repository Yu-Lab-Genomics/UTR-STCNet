#!/usr/bin/env python
# coding: utf-8
# import warnings
# warnings.filterwarnings('ignore')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" #  4, 5, 6

from UTR.utr_dataset_all import *

from UTR.UTRFormer import utrformer_large as utrformer
from util import ScheduledOptim

from copy import deepcopy
from tqdm import tqdm, trange
import torch
from torchsummary import summary
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
random.seed(1337)

import matplotlib.pyplot as plt

import argparse
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import CyclicLR


parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', type=str, default='0', help="Training Devices")
parser.add_argument('--local_rank', type=int, default=0, help="DDP parameter, do not modify")

parser.add_argument('--prefix', type=str, default = 'UTR')  # UTR 
parser.add_argument('--seq_type', type=str, default = 'utr')  # UTR 
parser.add_argument('--label-class', type=str, default = 'rl')  # rl 
parser.add_argument('--cell_line', type=str, default = '') 
parser.add_argument('--epochs', type = int, default = 200)
parser.add_argument('--layers', type = int, default = 16)  
parser.add_argument('--heads', type = int, default = 16)
parser.add_argument('--embed_dim', type = int, default = 128)
parser.add_argument('--batchsize', type = int, default = 256)
parser.add_argument('--batch-tokens', type = int, default = 12800)
parser.add_argument('--seq_max_len', type = int, default = 120)
parser.add_argument('--structure_weight', type = float, default = 1)
parser.add_argument('--train_file', type = str, required=True) 
parser.add_argument('--val_file', type = str, required=True) 
parser.add_argument('--lr', type = float, default = 1e-5)
parser.add_argument('--load_wholemodel', action = 'store_true') ## if --: True
parser.add_argument('--modelfile', type = str, default = '')
parser.add_argument('--init_epochs', type = int, default = 0)
parser.add_argument('--optimizer', type = str, default = "Schedule")
args = parser.parse_args()

global idx_to_tok, prefix, epochs, layers, heads, embed_dim, batch_toks, device, repr_layers, evaluation, include, truncate, return_contacts, return_representation, mask_toks_id ,device_ids, structure_tok_to_idx
print("*"*20, args.train_file)

device_ids = list(map(int, args.device_ids.split(',')))
device = torch.device('cuda:{}'.format(device_ids[args.local_rank]))
torch.cuda.set_device(device)


prefix = f'{args.prefix}_{args.label_class.upper()}'
epochs = args.epochs

outputfilename = f'{prefix}_epoch{args.epochs}_batchsize{args.batchsize}_padd{args.seq_max_len}'
print(outputfilename)

if args.val_file is None:
    train_data, val_data, seq_max_len = load_train_val_data(args.train_file, args.seq_type, val_size=0.1)
else:
    train_data, seq_max_len = load_train_data(args.train_file, seq="utr")
    val_data = load_test_data(args.val_file)

if args.seq_max_len == 0:
    args.seq_max_len = seq_max_len



train_datata = UTRDATA(train_data, args.seq_type, seqs_max_length=args.seq_max_len, label_class=args.label_class.lower())
val_datata = UTRDATA(val_data, args.seq_type, seqs_max_length=args.seq_max_len, label_class=args.label_class.lower())

train_batches_loader = torch.utils.data.DataLoader(train_datata, 
                                             batch_size = args.batchsize, # args.batchsize; 1
                                             num_workers = 8,
                                             shuffle = True
                                             )
val_batches_loader = torch.utils.data.DataLoader(val_datata, 
                                             batch_size = 2048, # args.batchsize; 1
                                             num_workers = 8,
                                             shuffle = True
                                             )

model = utrformer(padding_idx=train_datata.padding_idx, token_cls=train_datata.token_class, pooling_size = args.seq_max_len).to(device)



total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f"Total params    : {total_params:,}")
# print(f"Trainable params: {trainable_params:,}")
# print(f"≈ {total_params*4/1024**2:.2f} MB (float32)")




if args.load_wholemodel: 
    model.load_state_dict({k.replace('module.', ''):v for k,v in torch.load(args.modelfile)["model"].items()}) # , strict=False
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=device_ids)
    

optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                betas=(0.9, 0.98),
                                eps=1e-8)
# scheduler = CyclicLR(optimizer, base_lr=args.lr*0.1, max_lr=args.lr, step_size_up=200, mode='triangular')
scheduler=CyclicLR(optimizer, base_lr=args.lr, max_lr=args.lr*10, step_size_up=10, step_size_down=50)

loss_supervised_best, ep_best = np.inf,  -1
loss_supervised_list= []
criterion = torch.nn.MSELoss().to(device)
# criterion = nn.SmoothL1Loss(beta=1e-2).to(device)
# criterion = nn.HuberLoss(delta=1.0).to(device)

dir_saver = f'./checkpoint/{args.prefix}/'
if not os.path.exists(dir_saver):
    os.makedirs(dir_saver)

r2_best, spr_best, rmse_best = float('-inf'), float('-inf'), float('inf')


if not os.path.exists('./figures/'): os.makedirs('./figures/')
model.train()
for epoch in range(args.init_epochs+1, args.init_epochs + epochs + 1):
    loss_supervised_epoch = []
    with tqdm(total=len(train_batches_loader)) as t:
        for strs, tokens, labels  in train_batches_loader:
            t.set_description(f'Epoch {epoch} ')
            input = tokens.to(device)
            targets = torch.tensor(labels, dtype=torch.float).to(device)
            optimizer.zero_grad()
            out, out2 = model(input)
            loss1 = criterion(out.squeeze(), targets)
            loss = loss1
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
            for _, batch in enumerate(tqdm(val_batches_loader)):
                strs, tokens, labels = batch
                input = tokens.to(device)
                targets = torch.tensor(labels).to(device)
                out, _ = model(input)
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
        print(f">>>>> val-: val dataset R-squared: {round(r2, 4)}  Spearman R: {round(spearman_corr, 4)}   RMSE: {round(rmse, 4)} LR: {round(optimizer.param_groups[0]['lr'], 7)}\n") 

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
        if epoch > 10 and optimizer.param_groups[0]['lr']>1e-5:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1
    
    # if (loss_supervised_epoch < loss_supervised_best * 0.8 and epoch > 100) or epoch == args.epochs: 
    #     loss_supervised_best, ep_best = loss_supervised_epoch, epoch
    #     path_saver = dir_saver + f'{outputfilename}_MRLLossMin_epoch{epoch}.pkl'
    #     print(f'****Saving model in {path_saver}: \nBest epoch = {ep_best}')
    #     ckpt = {
    #         "model": model.state_dict(),
    #     }
    #     torch.save(ckpt, path_saver)


    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (15, 10))
    axes.plot(range(epoch-args.init_epochs), loss_supervised_list, label = 'Supervised Info|MSELoss')
    plt.title(f'{outputfilename}')
    plt.legend()
    plt.savefig(f'./figures/{outputfilename}.tif')

