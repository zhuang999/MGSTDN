import torch
import torch.nn as nn
from enum import Enum
import time
import numpy as np
from utils import *
import scipy.sparse as sp
import math
import matplotlib.pyplot as plt
import numpy as np
from mlp_ib import TriMixer, TriMixer_adj, MultiLayerPerceptron, MixerBlock
from model_attn import MultiHeadAttention

class GlobalKernelModule(nn.Module):
    def __init__(self, seq_len, eye_matrix=False):
        super(GlobalKernelModule, self).__init__()
        if eye_matrix:
            mask = torch.eye(seq_len)
        else:
            mask = torch.triu(torch.ones([seq_len, seq_len]))
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        self.kernel = nn.Parameter(matrix, requires_grad=True)

    def forward(self):
        # 可以根据需要定义前向计算
        return self.kernel

class trend_sample(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, hidden_size, seq_len, quadkey_level, setting):
        super(trend_sample, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleList([
                nn.Sequential(
                    torch.nn.Linear(i, i),
                    nn.GELU(),
                    torch.nn.Linear(i, 1),
                )
                for i in range(1, seq_len+1)
                ]) for _ in reversed(range(quadkey_level-1))
            ]
        )
        self.hidden_size = hidden_size
        self.quadkey_level = quadkey_level
        self.act = nn.Sigmoid()
        self.mlp = nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    torch.nn.Linear(hidden_size, hidden_size),
                )
        self.mixing_high_low = torch.nn.ModuleList( [MixerBlock(hidden_size) for _ in range(quadkey_level-1)] )  #GlobalKernelModule(seq_len)   torch.nn.Linear(hidden_size, hidden_size) MixerBlock(hidden_size)
        self.mixing_low_high = torch.nn.ModuleList(
                                [MixerBlock(hidden_size) for _ in range(quadkey_level-1)]
        )
        self.mixing_high_low_fine = torch.nn.ModuleList( [MixerBlock(hidden_size) for _ in range(quadkey_level-1)] ) 
        self.mixing_high_low_coarse = torch.nn.ModuleList( [MixerBlock(hidden_size) for _ in range(quadkey_level-1)] ) 
        self.mixing_low_high_coarse = torch.nn.ModuleList(
                                [MixerBlock(hidden_size) for _ in range(quadkey_level-1)]
        )
        self.mixing_low_high_fine = torch.nn.ModuleList(
                                [MixerBlock(hidden_size) for _ in range(quadkey_level-1)]
        )
        self.project = torch.nn.Linear(hidden_size, hidden_size)
        self.multihead_enc = MultiHeadAttention(setting.device, hidden_size, 5, 5, 4)
    def global_kernel(self, seq_len):
        mask = torch.triu(torch.ones([seq_len, seq_len]))
        matrix = torch.ones([seq_len, seq_len])
        matrix = matrix.masked_fill(mask == 0.0, -1e9)
        kernel = nn.parameter.Parameter(matrix, requires_grad=True)
        return kernel

    def forward(self, trend_list, season_list, enc_seq, step, attn_mask=None, state="train"):
        # if state == "train":
        #mixing high->low  本体映射
        trend_high_low_list = trend_list.copy()
        trend_high_low_list.reverse()
        out_high = trend_high_low_list[0]
        out_low = trend_high_low_list[1]
        high_low_list = [out_high]
        for i in range(self.quadkey_level-1):
            high_low_res = self.mixing_high_low[i](out_high, None)
            #high_low_res = self.act(torch.matmul(out_high.permute(0, 2, 1), self.mixing_high_low[i]().softmax(dim=-1)).permute(0, 2, 1))
            # 计算门控值
            # gate_value = torch.sigmoid(self.mixing_high_low[i](out_high))
            # # 变换细粒度和粗粒度信息
            # fine_transformed = torch.tanh(self.mixing_high_low_coarse[i](out_low))
            # coarse_transformed = torch.tanh(self.mixing_high_low_fine[i](out_high))
            # # 门控单元输出
            # out_low = gate_value * fine_transformed + (1 - gate_value) * coarse_transformed
            if i != 0:
                high_low_res = high_low_res + enc_seq
            out_low = out_low + high_low_res
            out_high = out_low
            if i + 2 <= self.quadkey_level - 1:
                out_low = trend_high_low_list[i + 2]
            high_low_list.append(out_high)
        # else:
        #     high_low_list = trend_list.copy()
        #     high_low_list.reverse()
        
        # high_low_season_list = []
        # high_low_reverse = high_low_list.copy()
        # high_low_reverse.reverse()
        # for i in range(len(high_low_reverse) - 1):
        #     trend = high_low_reverse[i]
        #     season = high_low_reverse[i+1] - trend
        #     high_low_season_list.append(season)
        

        # mixing low->high
        low_high_list = high_low_list.copy()
        low_high_list.reverse()  #漏了
        out_low = low_high_list[0]
        out_high = low_high_list[1]
        out_trend_list = [out_low+enc_seq]#+step[:,:,0]]  #+high_low_list[0]
        for i in range(self.quadkey_level - 1):
            # gate_value = torch.sigmoid(self.mixing_low_high[i](out_low))
            
            # # 变换细粒度和粗粒度信息
            # fine_transformed = torch.tanh(self.mixing_low_high_coarse[i](out_high))
            # coarse_transformed = torch.tanh(self.mixing_low_high_fine[i](out_low))
            # # 门控单元输出
            # out_high = gate_value * fine_transformed + (1 - gate_value) * coarse_transformed

            out_low = self.mixing_low_high[i](out_low, None)
            # #low_high_res = self.act(torch.matmul(x.permute(0, 2, 1), self.mixing_low_high[i]().softmax(dim=-1)).permute(0, 2, 1))
            out_high = out_high + out_low #+ step[:,:,i+1] #+ high_low_list[i+1]
            enc_seq = self.project(enc_seq)
            if i < self.quadkey_level - 2:
                out_high = out_high + enc_seq #self.multihead_enc(enc_seq, out_high, out_high, attn_mask) #+ high_low_season_list[i]

            out_low = out_high
            if i + 2 <= self.quadkey_level - 1:
                out_high = low_high_list[i + 2]
            out_trend_list.append(out_low)
        high_low_list.reverse()
        return out_trend_list, high_low_list


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, hidden_size, seq_len, quadkey_level):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList([
                torch.nn.ModuleList([
                nn.Sequential(
                    torch.nn.Linear(i, i),
                    nn.GELU(),
                    torch.nn.Linear(i, 1),
                )
                for i in range(1, seq_len+1)
                ]) for _ in range(quadkey_level-1)
            ]
        )
        self.hidden_size = hidden_size

        self.mlp = nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    torch.nn.Linear(hidden_size, hidden_size),
                )


    def forward(self, season_list):
        #expect shape: [B, N, D]
        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        user_len, seq_len, _ = out_high.shape
        out_season_list = [out_high]
        for i in range(len(season_list) - 1):
            out_low_res = torch.zeros(user_len, seq_len,
                            self.hidden_size, device=out_high.device)
            for j in range(seq_len):
                out_low_res[:, j, :] = self.down_sampling_layers[i][j](out_high[:, 0:j+1].permute(0, 2, 1)).squeeze(-1)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_high = self.mlp(out_high)
            out_season_list.append(out_high)

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, hidden_size, seq_len, quadkey_level):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleList([
                nn.Sequential(
                    torch.nn.Linear(i, i),
                    nn.GELU(),
                    torch.nn.Linear(i, 1),
                )
                for i in range(1, seq_len+1)
                ]) for _ in reversed(range(quadkey_level-1))
            ]
        )
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    torch.nn.Linear(hidden_size, hidden_size),
                )


    def forward(self, trend_list):
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        user_len, seq_len, _ = out_high.shape
        out_trend_list = [out_low]
        for i in range(len(trend_list_reverse) - 1):
            out_high_res = torch.zeros(user_len, seq_len,
                            self.hidden_size, device=out_high.device)
            for j in range(seq_len):
                out_high_res[:, j] = self.up_sampling_layers[i][j](out_low[:, :j+1].permute(0, 2, 1)).squeeze(-1)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_low = self.mlp(out_low)
            out_trend_list.append(out_low)
        out_trend_list.reverse()
        return out_trend_list