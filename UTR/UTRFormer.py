
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
            sr_ratios=[8, 4, 2, 1], num_stages=4, pretrained=None, k=3, sample_ratios=[0.8, 0.5, 0.5], # 1.[0.25, 0.25, 0.25] MRL:[0.8, 0.5, 0.2] other:[0.5, 0.5, 0.2]
            return_map=False, **kwargs
    ):
        super().__init__()

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

        for i in range(1):
            # patch_embed = TokenEmbed(embed_dim=embed_dims[i], **kwargs)
            patch_embed = nn.Sequential(
                        nn.Linear(4, embed_dims[i]),
                        norm_layer(embed_dims[i]),
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
                    nn.Linear(64, 1)
                )
        # self.head = nn.Linear(512, 1)
        
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

    def forward(self, x, train=True):
        x0 = self.forward_features(x)
        x_s = x0[-1]['x']

        # x_s_idx_token = x0[-1]['idx_token']
        # x_s_agg_weight = x0[-1]['agg_weight'].squeeze()
        # x_s_agg_weight_normalized = F.normalize(x_s_agg_weight, p=2, dim=1)
        # x_s_idx_token_embed = self.embed_class(x_s_idx_token)
        
        # norms = torch.norm(x_s, dim=2, keepdim=True)  # shape: (batch_size, 1, num_cols)
        # normalized_batch_matrix = x_s / norms
        # cosine_similarity = torch.bmm(normalized_batch_matrix, normalized_batch_matrix.transpose(1, 2))  # shape: (32, 5, 5)
        # batch_size, n_rows, _ = cosine_similarity.shape
        # mask = torch.eye(n_rows, device=cosine_similarity.device).expand(batch_size, -1, -1)
        # cosine_similarity = cosine_similarity * (1 - mask)
        # print("Modified Cosine Similarity Matrix Shape:", cosine_similarity.shape)
        
        # x = x0[-1]['x'].mean(dim=1)
        # x = self.pooling(x0[-1]['x'].transpose(1, 2)).squeeze()
        
        x = self.pooling(x0[-1]['x'])
        
        # output_sequence, (final_hidden_state_1, final_hidden_state_2) = self.head_gru(x_s)
        # x = self.head_re(final_hidden_state_2)
        x = self.head(x)
        if train:
            return x, x_s
        else:
            return x0, x
    

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
        
