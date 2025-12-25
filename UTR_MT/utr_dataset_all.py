from tkinter import NO
from typing import List
import math
import random
from copy import deepcopy
from typing import Sequence, Tuple, List, Union
import itertools
from sympy import N
import torch
from typing import Any, Callable, Optional, Tuple, List
import torch.utils.data as data
import os
import numpy as np
import pandas as pd
import tqdm


class BatchConverter(object):
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]): 
        batch_size = len(raw_batch)
        seq_list, seq_encoded_list, batch_labels = zip(*raw_batch)
        assert len(set(seq_list)) == len(seq_list)
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = torch.empty((batch_size,max_len),dtype=torch.int64,)
        tokens.fill_(self.padding_idx)
    
        labels = []
        seq_strs = []
        for i, (label, seq_encoded, seq_str) in enumerate(zip(batch_labels, seq_encoded_list, seq_list)):
            labels.append(label) 
            seq_strs.append(seq_str)            
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i,  : len(seq_encoded)] = seq
            
        return labels, tokens, seq_strs


class RawData():
    def __init__(self, fname, seq="utr", label="rl"):
        self.df = pd.read_csv(fname)
        self.input_seq = seq
        self.label = label
        self.df_data = self.df.loc[:, [self.input_seq, self.label]].replace('<pad>', 'P', regex=True)
        self.get_seq_map()
        self.seqs_max_length = self.get_seqs_max_length()
        print(f">>>>>>>Input info: Input {self.input_seq}, Label: {self.label}, Map_dict: {self.seq_map}")
    
    def __len__(self):
        return len(self.df_data)
    
    def get_seq_map(self):
        if "utr" in self.input_seq:
            if "rl" in self.label:
                self.seq_map = {
                "A":1,
                "C":2,
                "G":3,
                "T":4,
                "N":0,
                }
                self.embed_num = 5
                self.padding_idx = 0
            else:
                self.seq_map = {
                    "A":1,
                    "C":2,
                    "G":3,
                    "T":4,
                    "N":0
                }
                self.embed_num = 5
                self.padding_idx = 0
        elif self.input_seq == "ss":
            self.seq_map = {
                "(":1,
                ".":2,
                ")":3,
                "P":0
            }
            self.embed_num = 4
            self.padding_idx = 0
        else:
            self.seq_map = None
            self.embed_num = 0
            self.padding_idx = 0

    def get_df(self):
        return self.df_data

    def get_seqs(self):
        return self.df_data[self.input_seq].to_list()
    
    def get_seqs_max_length(self):
        return max(self.df_data[self.input_seq].str.len())

    def get_labels(self):
        return self.df_data[self.label]
    
    def encoder(self):
        labels = self.get_labels()
        seq_str = []
        encoder_seq = []
        seq_label = []
        print("正在处理数据。。。。")
        for idx, sub_seq in enumerate(self.get_seqs()):
            if sub_seq in seq_str:
                continue
            seq_label.append(labels[idx])
            seq_str.append(sub_seq)
            encoder_seq.append([self.seq_map[item] for item in sub_seq])
        print("处理数据结束。")
        return seq_str, encoder_seq, seq_label


class UTRDATA(data.Dataset):
    def __init__(self,root: str, seq_type: str,label_class: Optional[Callable] ="rl", train=True):  # "te", "rpkm_rnaseq"
        super().__init__()
        self.seq=seq_type
        self.train = train
        self.label_class = label_class
        self.rawdata = RawData(root, seq=self.seq, label=self.label_class)
        print(">>>> 数据集大小： ", len(self.rawdata))
        self.seq_max_length = self.rawdata.seqs_max_length
        # self.seq_max_length = 105
        # self.data, self.encoder_data, self.target = self.rawdata.encoder()
        self.data = self.rawdata.get_df()
        self.token_class = self.rawdata.embed_num 

    def __getitem__(self, index: int):
        return self.encoder(index)

    def __len__(self) -> int:
        return len(self.data)
 
    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.data)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0
        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0
        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)
        _flush_current_buf()
        return batches
    
    def get_batch_converter(self):
        return BatchConverter(padding_idx=self.rawdata.padding_idx)
    
    def encoder(self, idx):
        seq_str = self.data.iloc[idx][self.seq]
        if "label" in self.label_class:
            seq_label = int(self.data.iloc[idx][self.label_class])
        else:
            seq_label = self.data.iloc[idx][self.label_class]
        if self.train:
            tokens = torch.empty(self.seq_max_length,dtype=torch.int64)
        else:
            tokens = torch.empty(self.seq_max_length,dtype=torch.int64)
        tokens.fill_(0)
        encoder_seq = torch.tensor([self.rawdata.seq_map[item] for item in seq_str], dtype=torch.int64)
        tokens[:len(encoder_seq)]= encoder_seq
        return seq_str, tokens, seq_label

# file = '/pool1/liuzhouwu/datasets/UTR/muscle_merged_traintData.csv'
# dataset = UTRDATA(root=file, utr="UTR", label_class="te")
# batches = dataset.get_batch_indices(toks_per_batch = 1024, extra_toks_per_seq = 0)
# batches_loader = torch.utils.data.DataLoader(batches, 
#                                              batch_size = 1,
#                                              num_workers = 8,
#                                              shuffle = True
#                                              )
# for i, batch in enumerate(batches_loader):
#     batch = np.array(torch.LongTensor(batch)).tolist()
#     dataloader = torch.utils.data.DataLoader(dataset, 
#                                             collate_fn=dataset.get_batch_converter(), 
#                                             batch_sampler=[batch], 
#                                             shuffle = False)
#     for (labels, tokens, strs) in dataloader:
#         print(len(labels))
#         # print(labels[0])
#         # print(tokens[0])
#         # print(strs[0])
