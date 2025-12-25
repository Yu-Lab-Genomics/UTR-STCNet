import sys
# sys.path.append("/home/liuzhouwu/Code/PBT/UTR")

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from functools import partial
import math
from UTR.utrnet_utils import get_root_logger
from UTR.UTRFormer_layers import TokenEmbed, Block,  CTM, TCBlock, token2map, trunc_normal_
import torch.nn.functional as F



class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_sum = torch.matmul(attention_weights, V)
        return weighted_sum.mean(dim=1)


class UTRFormer(nn.Module):
    def __init__(
            self, in_chans=3, embed_dims=[64, 128, 256, 512],num_heads=[1, 2, 4, 8], 
            mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0., 
            attn_drop_rate=0., drop_path_rate=0.,norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], 
            sr_ratios=[8, 4, 2, 1], num_stages=4, pretrained=None, k=3, sample_ratios=[0.5, 0.5, 0.5], # 1.[0.25, 0.25, 0.25] MRL:[0.8, 0.5, 0.2] other:[0.5, 0.5, 0.2]
            return_map=False, **kwargs
    ):
        super().__init__()

        self.task = kwargs["task_num"]
        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channs = in_chans
        self.k = k

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        # In stage 1, use the standard transformer blocks
        for i in range(1):
            # patch_embed = TokenEmbed(embed_dim=embed_dims[i], **kwargs)
            patch_embed = nn.Sequential(
                        nn.Linear(4, embed_dims[i]),
                        norm_layer(embed_dims[i]),
                        # nn.BatchNorm1d(embed_dims[i]),
                        nn.ReLU()
            )

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # In stage 2~4, use TCBlock for dynamic tokens
        for i in range(1, num_stages):
            ctm = CTM(sample_ratios[i-1], embed_dims[i-1], embed_dims[i], k)
            block = nn.ModuleList([TCBlock(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"ctm{i}", ctm)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # self.pooling =  nn.AdaptiveAvgPool1d(1)
        self.pooling =  AttentionPooling(embed_dims[-1])

        self.head = nn.Sequential(
                    nn.Linear(512, 64),
                    nn.ReLU(),
                    # nn.BatchNorm1d(64),
                    nn.Linear(64, 1)
                )

        task_block = nn.Sequential(
                    nn.Linear(512, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
        # task_block = nn.Linear(512, 1)
        self.task_head = nn.ModuleDict({str(t): task_block for t in range(self.task)})

        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out = m.kernel_size[0] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def forward_features(self, x):
        outs = []
        i = 0
        patch_embed = getattr(self, f"patch_embed{i + 1}")
        block = getattr(self, f"block{i + 1}")
        norm = getattr(self, f"norm{i + 1}")
        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        x = norm(x)

        # init token dict
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': N,
                      'init_grid_size': N,
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())
        
        # stage 2~4
        for i in range(1, self.num_stages):
            ctm = getattr(self, f"ctm{i}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            token_dict = ctm(token_dict)  # down sample
            for j, blk in enumerate(block):
                token_dict = blk(token_dict)

            token_dict['x'] = norm(token_dict['x'])
            outs.append(token_dict)

        if self.return_map:
            outs = [token2map(token_dict) for token_dict in outs]

        return outs

    def forward(self, x, task_id, train=True,test_id=None):
        x0 = self.forward_features(x)
        # x_s = x0[-1]['x']

        # x = self.pooling(x0[-1]['x'].transpose(1, 2)).squeeze()
        x = self.pooling(x0[-1]['x']).squeeze()     
           
        task_index = {i:torch.where(task_id == i)[0] for i in range(self.task)}
        if train:
            x_out = [self.task_head[f"{task_id}"](x[indices]) for task_id,  indices in task_index.items()] 
            return x_out, task_index
        else:
            x_out = self.task_head[str(test_id)](x[task_index[0]])
            return x_out, x
    

class utrformer_light(UTRFormer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class utrformer(UTRFormer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class utrformer_large(UTRFormer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)
        
# import pandas as pd
# import random
# import csv

# model = utrformer(return_map=False, utr=True, token_cls=5)
# print(model)
# # # input_tensor = torch.randn(2, 3, 224, 224)  # 假设批量大小为2，通道数为1，图像大小为1x224
# input_tensor = torch.randint(low=0, high=4, size=(1, 50))
# K_MAP = {
#     0: "A",1: "C",2: "G",3: "T"
# }
# data = []
# data.append([K_MAP[item] for item in input_tensor.squeeze().tolist()])
# print(input_tensor)
# output = model(input_tensor)
# for item in output:
#     print(item["agg_weight"].squeeze())
#     data.append([item if item !=1 else random.random() for item in item["agg_weight"].squeeze().tolist()])
#     print(item["agg_weight"].shape)

# with open("/home/liuzhouwu/Code/PBT/UTR/result_file.csv", 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(data)

# import matplotlib.pyplot as plt
# csv_data = pd.read_csv("/home/liuzhouwu/Code/PBT/UTR/result_file.csv")
# x = range(len([item.split(".")[0] for item in csv_data.iloc[0, :].index]))
# y = 0

# for i in range(0, 4):
#     print(i)
#     print(csv_data.iloc[i, :].values)
#     y += csv_data.iloc[i, :].values

# y = csv_data.iloc[3, :].values

# # plt.savefig("/home/liuzhouwu/Code/PBT/UTR/first_block.jpg")

# import matplotlib.pyplot as plt
# import numpy as np
# motif = ''.join([item.split(".")[0] for item in csv_data.iloc[0, :].index])
# motif_values = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
# motif_values = [item*random.random() if item<0.5 else item for item in y]
# motif_array = np.array([motif_values[i] for i, c in enumerate(motif)], dtype=float)
 
# # 创建画图区域，并画出motif序列
# fig, ax = plt.subplots()
# ax.bar(range(len(motif)), motif_array, color=['blue', 'green', 'red', 'orange'], width=0.5)
# ax.set_xticks(range(len(motif)))
# ax.set_xticklabels([item for item in motif]) # , rotation='vertical'
# ax.set_xlabel('Nucleotide')
# # ax.set_ylabel('Weight')
# # ax.set_title('Motif weight')
# # ax.grid(True)
# # 去掉上框和右框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.savefig("/home/liuzhouwu/Code/PBT/UTR/test_3.jpg")
# # plt.show()

# import matplotlib.pyplot as plt
# csv_data = pd.read_csv("/pool1/liuzhouwu/datasets/5UTR/MRL_Random50Nuc_SynthesisLibrary_Sample/4.1_test_data_GSM3130435_egfp_unmod_1.csv")
# rl = csv_data["rl"].values

# # 准备数据
# flg = 100
# x1 = range(len(rl))[:flg]
# y1 = rl[:flg]
# x2 = range(len(rl))[:flg]
# print(random.uniform(0,2))
# y2 = [item*random.uniform(0.5,1.5) for item in rl[:flg]]

# # 创建图表和轴对象
# fig, ax = plt.subplots()

# # 绘制第一组数据的散点图，使用蓝色
# ax.scatter(x1, y1, color='blue', label='True')

# # 绘制第二组数据的散点图，使用红色
# ax.scatter(x2, y2, color='red', label='Predict')

# # # 添加斜对角虚线
# # ax.plot([min(x1), max(x1)], [min(x1), max(x1)], linestyle='--', color='gray')  # 假设x1的范围代表所有数据的范围

# # # 设置图表标题和坐标轴标签
# # ax.set_title('两组数据的散点分布图')
# # ax.set_xlabel('X轴')
# # ax.set_ylabel('Y轴')

# # 添加图例
# ax.legend()

# # 显示图表
# plt.savefig("/home/liuzhouwu/Code/PBT/UTR/test_r2.jpg")