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
import torch.nn.functional as F

from mlp_ib import TriMixer, TriMixer_adj, MultiLayerPerceptron, MixerBlock
#from DualTransformer import DualInfoTransformer
from attention_aggregator import attention_aggregator2
from model_attn import MultiHeadAttention
from model_geo import GeoEncoderLayer, GeoEncoder, EncoderLayer, SASEncoder, DecoderLayer, Decoder
# from layers.StandardNorm import Normalize
from multiscale_sample import trend_sample
from multiscale_sample import GlobalKernelModule

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class PastDecomposableMixing(nn.Module):
    def __init__(self, hidden_size, seq_len, quadkey_level, setting):
        super(PastDecomposableMixing, self).__init__()

        self.cross_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
        )

        # Mixing season
        self.mixing_trend_sample = trend_sample(hidden_size, seq_len, quadkey_level, setting)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size*2),
        )

    def forward(self, x_list, enc_seq, step, attn_mask, state):
        # expect shape: [B, N, D]
        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for i in range(len(x_list) - 1):
            trend = x_list[i]
            season = x_list[i+1] - trend
            # if self.channel_independence == 0:
            #     season = self.cross_layer(season)
            #     trend = self.cross_layer(trend)
            season_list.append(season)
            trend_list.append(trend)
        trend_list.append(x_list[-1])

        out_list, out_list2 = self.mixing_trend_sample(trend_list, season_list, enc_seq, step, attn_mask, state)  # top-down-sample

        # output_list = []
        # for i, out_trend in zip(range(len(x_list)), out_list): #相加位置尝试变化一下
        #     out = self.out_cross_layer(out_trend)
        #     output_list.append(out)
        return out_list, out_list2


class Rnn(Enum):
    """ The available RNN units """

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        raise ValueError('{} not supported in --rnn'.format(name))


class RnnFactory():
    """ Creates the desired RNN unit. """

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size) 
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)

        
class Flashback(nn.Module):
    """ Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    """

    def __init__(self, input_size, user_count, quadkey_count, quadkey_location, time_count, hidden_size, f_t, f_s, rnn_factory, lambda_loc, lambda_user, use_weight,
                 graph, spatial_graph, friend_graph, use_graph_user, use_spatial_graph, interact_graph, setting):
        super().__init__()
        self.input_size = input_size  # POI个数
        self.user_count = user_count
        self.quadkey_count = quadkey_count
        self.quadkey_location = quadkey_location
        self.time_count = time_count
        self.add_input_size(input_size)
        self.hidden_size = hidden_size
        self.f_t = f_t  # function for computing temporal weight
        self.f_s = f_s  # function for computing spatial weight
        self.user_len = setting.batch_size
        self.seq_len = setting.sequence_length

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight
        self.use_graph_user = use_graph_user
        self.use_spatial_graph = use_spatial_graph
        self.map_level = len(setting.quadkey_list)
        self.quadkey_num = len(setting.quadkey_list) + 1

        self.I = identity(graph.shape[0], format='coo')
        self.graph = sparse_matrix_to_tensor(
            calculate_random_walk_matrix((graph * self.lambda_loc + self.I).astype(np.float32)))

        self.spatial_graph = spatial_graph
        if interact_graph is not None:
            self.interact_graph = sparse_matrix_to_tensor(calculate_random_walk_matrix(
                interact_graph))  # (M, N)
        else:
            self.interact_graph = None

        self.encoder = nn.Embedding(
            input_size, hidden_size)  # location embedding
        self.quad_encoder = nn.ModuleList([nn.Embedding(count_num, hidden_size * 2) for count_num in quadkey_count])
        # self.encoder.load_state_dict(torch.load('embedding_checkpoint_3_1.pth'))
        # self.time_encoder = nn.Embedding(24 * 7, hidden_size)  # time embedding
        self.time_encoder = nn.ModuleList([nn.Embedding(count_num, hidden_size * 2) for count_num in time_count])
        self.step_encoder = nn.Embedding(self.quadkey_num, hidden_size * 2)
        self.user_encoder = nn.Embedding(
            user_count, hidden_size)  # user embedding
        self.rnn = rnn_factory.create(hidden_size)
        self.fc = nn.Linear(2 * hidden_size, input_size)
        self.fc0 = nn.ModuleList([nn.Linear(hidden_size * 2, count_num) for count_num in quadkey_count])
        self.fc1 = nn.ModuleList([nn.Linear(hidden_size * 2, count_num) for count_num in quadkey_count])
        self.fc2 = nn.ModuleList([nn.Linear(hidden_size * 2, count_num) for count_num in time_count])
        #self.mlpmixer = MixerBlock(hidden_size)
        self.globalkernel = GlobalKernelModule(self.seq_len)
        #self.dualtrans = DualInfoTransformer()

        self.layer = 1
        self.quad_embedding = nn.Linear(1, hidden_size)
        
        self.pdm_blocks_quad = nn.ModuleList([PastDecomposableMixing(hidden_size * 2, self.seq_len, self.quadkey_num, setting)
                                         for _ in range(self.layer)])
        self.pdm_blocks_time = nn.ModuleList([PastDecomposableMixing(hidden_size * 2, self.seq_len, len(self.time_count), setting)
                                         for _ in range(self.layer)])
        self.predict_layers = torch.nn.ModuleList(
                [torch.nn.Linear(hidden_size, hidden_size) for i in range(self.quadkey_num)]
            )
        self.out_res_layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleList([
                nn.Sequential(
                    torch.nn.Linear(i, i),
                    nn.GELU(),
                    torch.nn.Linear(i, i),
                )
                for i in range(1, self.seq_len+1)
                ]) for _ in range(self.quadkey_num-1)
            ]
        )

        self.classifier_layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleList([
                    torch.nn.Linear(i, 1) for i in range(1, self.seq_len+1)
                ]) for _ in range(self.quadkey_num-1)
            ]
        )

        self.projection_layer = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm1 = nn.LayerNorm(self.seq_len)
        self.norm2 = nn.LayerNorm(hidden_size)

        depth = 1
        exp_factor = 4
        dropout = 0.5
        # self.position_embeddings = nn.Embedding(100, hidden_size)
        self.enc_layer = EncoderLayer(hidden_size, exp_factor, dropout)
        self.enc = SASEncoder(hidden_size, self.enc_layer, depth)

        self.multihead_enc = MultiHeadAttention(setting.device, hidden_size, 5, 5, 4)

        self.attn_head_quadkey = torch.nn.ModuleList(
            [
            MultiHeadAttention(setting.device, hidden_size * 2, 5, 5, 4) for _ in range(self.quadkey_num)
            ]
            )
        self.attn_head_time = torch.nn.ModuleList(
            [
            MultiHeadAttention(setting.device, hidden_size * 2, 5, 5, 4) for _ in range(len(self.time_count))
            ]
            )
        self.cross_attn_quad = MultiHeadAttention(setting.device, hidden_size * 2, 5, 5, 4)
        self.cross_attn_time = MultiHeadAttention(setting.device, hidden_size * 2, 5, 5, 4)
        self.cross_attn_quad_poi = MultiHeadAttention(setting.device, hidden_size * 2, 5, 5, 4)
        self.cross_attn_time_poi = MultiHeadAttention(setting.device, hidden_size * 2, 5, 5, 4)
        self.multi_mixing = torch.nn.ModuleList(
            [
                GlobalKernelModule(self.seq_len) for _ in range(self.quadkey_num)
            ]
            )
        self.act = nn.Sigmoid()
        self.project_out = nn.Linear(2 * hidden_size, hidden_size)
        self.project = nn.Linear(hidden_size, hidden_size)
        self.alpha = nn.Parameter(torch.FloatTensor(hidden_size*2))
        stdv = 1. / math.sqrt(self.alpha.shape[0])
        self.alpha.data.uniform_(-stdv, stdv)
        self.mlpmixer = MixerBlock(hidden_size)
        

    def add_input_size(self, input_size):
        self.quadkey_count.append(input_size)
        self.time_count.append(input_size)

    def Loss_l2(self):
        base_params = dict(self.named_parameters())
        loss_l2=0.
        count=0
        for key, value in base_params.items():
            if 'bias' not in key and 'pre_model' not in key:
                loss_l2+=torch.sum(value**2)
                count+=value.nelement()
        return loss_l2
    

    def forward(self, x, q_list, t, w, d, t_slot, hour, s, y_t, y_t_slot, y_s, h, active_user, epoch, state):
        #self.plt_fig(active_user, q_list)

        seq_len, user_len = x.size()
        x_emb = self.encoder(x)
        # # self.plt_fig(q)
        # 是否用GCN来更新user embedding
        if self.use_graph_user:
            # I_f = identity(self.friend_graph.shape[0], format='coo')
            # friend_graph = (self.friend_graph * self.lambda_user + I_f).astype(np.float32)
            # friend_graph = calculate_random_walk_matrix(friend_graph)
            # friend_graph = sparse_matrix_to_tensor(friend_graph).to(x.device)
            friend_graph = self.friend_graph.to(x.device)
            # AX
            user_emb = self.user_encoder(torch.LongTensor(
                list(range(self.user_count))).to(x.device))
            user_encoder_weight = torch.sparse.mm(friend_graph, user_emb).to(
                x.device)  # (user_count, hidden_size)

            if self.use_weight:
                user_encoder_weight = self.user_gconv_weight(
                    user_encoder_weight)
            p_u = torch.index_select(
                user_encoder_weight, 0, active_user.squeeze())
        else:
            p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
            # (user_len, hidden_size)
            p_u = p_u.view(user_len, self.hidden_size)

        p_u = self.user_encoder(active_user)  # (1, user_len, hidden_size)
        p_u = p_u.view(user_len, self.hidden_size)
        # AX,即GCN
        graph = self.graph.to(x.device)
        loc_emb = self.encoder(torch.LongTensor(
            list(range(self.input_size - 1))).to(x.device))
        encoder_weight = torch.sparse.mm(graph, loc_emb).to(
            x.device)  # (input_size, hidden_size)
        encoder_weight = torch.cat([encoder_weight, self.encoder.weight[-1:]],dim=0) # (input_size, hidden_size)
        
        if self.use_spatial_graph:
            spatial_graph = (self.spatial_graph *
                             self.lambda_loc + self.I).astype(np.float32)
            spatial_graph = calculate_random_walk_matrix(spatial_graph)
            spatial_graph = sparse_matrix_to_tensor(
                spatial_graph).to(x.device)  # sparse tensor gpu
            encoder_weight += torch.sparse.mm(spatial_graph,
                                              loc_emb).to(x.device)
            encoder_weight /= 2  # 求均值
       
        new_x_emb = []
        for i in range(seq_len):
            # (user_len, hidden_size)
            temp_x = torch.index_select(encoder_weight, 0, x[i])
            new_x_emb.append(temp_x)
        x_emb = torch.stack(new_x_emb, dim=0)  

        # user-poi
        loc_emb = self.encoder(torch.LongTensor(
            list(range(self.input_size - 1))).to(x.device))
        encoder_weight = loc_emb
        interact_graph = self.interact_graph.to(x.device)
        encoder_weight_user = torch.sparse.mm(
            interact_graph, encoder_weight).to(x.device)

        user_preference = torch.index_select(
            encoder_weight_user, 0, active_user.squeeze()).unsqueeze(0)
        # print(user_preference.size())
        user_loc_similarity = torch.exp(
            -(torch.norm(user_preference - x_emb, p=2, dim=-1))).to(x.device)
        user_loc_similarity = user_loc_similarity.permute(1, 0)

        attn_shape = (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0)
        subsequent_mask = subsequent_mask.long()
        attn_mask = subsequent_mask.to(x_emb.device)  #((1.0 - subsequent_mask) * (-2 ** 32 + 1)).to(x.device)
        out_w = self.multihead_enc(x_emb.transpose(0,1), x_emb.transpose(0,1), x_emb.transpose(0,1), attn_mask).transpose(0,1)
        out_w = self.mlpmixer(out_w, None)

        #out_w = self.act(torch.matmul(x_emb.permute(1, 2, 0), self.globalkernel().softmax(dim=-1)).permute(2, 0, 1))
        # out_w = torch.zeros(seq_len, user_len,
        #                     self.hidden_size, device=x.device)
        
        # for i in range(seq_len):
        #     sum_w = torch.zeros(user_len, 1, device=x.device)  # (200, 1)
        #     for j in range(i + 1):
        #         dist_t = t[i] - t[j]
        #         dist_s = torch.norm(s[i] - s[j], dim=-1)
        #         a_j = self.f_t(dist_t, user_len)  # (user_len, )
        #         b_j = self.f_s(dist_s, user_len)
        #         a_j = a_j.unsqueeze(1)  # (user_len, 1)
        #         b_j = b_j.unsqueeze(1)
        #         w_j = a_j * b_j + 1e-10  # small epsilon to avoid 0 division
        #         w_j = w_j * user_loc_similarity[:, j].unsqueeze(1)  # (user_len, 1)
        #         sum_w += w_j
        #         out_w[i] += w_j * enc_seq[j]  # (user_len, hidden_size)
        #     out_w[i] /= sum_w
            
        out_pu = torch.zeros(seq_len, user_len, 2 *
                             self.hidden_size, device=x.device)
        out_input = torch.zeros(seq_len, user_len, 2 *
                             self.hidden_size, device=x.device)
        for i in range(seq_len):
            # (user_len, hidden_size * 2)
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)
            out_input[i] = torch.cat([x_emb[i], p_u], dim=1)
        # out_pu = self.project_out(out_pu)

        step_list = torch.arange(0, self.quadkey_num).view(1,1,self.quadkey_num).repeat(user_len, seq_len, 1).to(x.device)
        step = self.step_encoder(step_list)


    # region multiscale regression 
        # embedding
        q_list = q_list.permute(1, 0, 2)#.float()
        #q_list = self.norm1(q_list.permute(0,2,1)).permute(0,2,1)
        quad_emb_list = []
        # season_list, trend_list = self.pre_enc(q_list)
        for i in range(self.quadkey_num-1): #-1
            enc_out = self.quad_encoder[i](q_list[..., i])   #.unsqueeze(-1)
            quad_emb_list.append(enc_out)
        # enc_out_list = []
        # for i in range(self.quadkey_num):  #-1
        #     #enc_out_list.append(self.act(torch.matmul(quad_emb_list[i].permute(0, 2, 1), self.multi_mixing[i]().softmax(dim=-1)).permute(0, 2, 1)))
        #     enc_out_list.append(self.attn_head_quadkey[i](quad_emb_list[i], quad_emb_list[i], quad_emb_list[i], attn_mask))
        quad_emb_list.append(out_input.transpose(0,1))

        # Past Decomposable Mixing as encoder for past
        #for i in range(self.layer):
        enc_out_list, enc_out_list2 = self.pdm_blocks_quad[0](quad_emb_list, out_pu.transpose(0,1), step, attn_mask, state)

        h_list =[timekey.permute(1, 0) for timekey in [w, d, t_slot, hour]]
        time_emb_list = []
        for i in range(len(self.time_count)-1):  #-1
            time_out = self.time_encoder[i](h_list[i].long())
            time_emb_list.append(time_out)

        # enc_time_out_list = []
        # for i in range(len(self.time_count)): #-1
        #     enc_time_out_list.append(self.attn_head_time[i](time_emb_list[i], time_emb_list[i], time_emb_list[i], attn_mask))
        time_emb_list.append(out_input.transpose(0,1))
        time_out_list, _ = self.pdm_blocks_time[0](time_emb_list, out_pu.transpose(0,1), step, attn_mask, state)


        # Future Multipredictor Mixing as decoder for future
        #dec_out_list = self.future_multi_mixing(enc_out_list)

        # query_out = torch.stack(enc_out_list, dim=-1).sum(-1).permute(1,0,2)
        # query_out = self.norm2(query_out)
        #dec_out = torch.stack(enc_out_list, dim=-1)#.sum(-1).squeeze(-1)  #.squeeze(-1)   #.permute(1,0,2)

        # hidden_state = self.mlpmixer(x_emb)
        #enc_seq = self.enc(x_emb.transpose(0,1), x_emb.transpose(0,1), x_emb.transpose(0,1),  None, attn_mask).transpose(0,1) 


        # out, h = self.rnn(x_emb, h)  # (seq_len, user_len, hidden_size)
        # #out_w = self.mlpmixer(x_emb)
        quad_out_list = torch.stack(enc_out_list, dim=-2).transpose(0,1)#.sum(-1).permute(1,0,2)
        quad_out_list2 = torch.stack(enc_out_list2, dim=-2).transpose(0,1)
        time_out_list = torch.stack(time_out_list, dim=-2).transpose(0,1)
        quad_out_list_ = quad_out_list.contiguous().view(-1, quad_out_list.shape[2], quad_out_list.shape[3])
        time_out_list_ = time_out_list.contiguous().view(-1, time_out_list.shape[2], time_out_list.shape[3])
        
        
        quad_out_list = self.cross_attn_quad(quad_out_list_, time_out_list_, time_out_list_).contiguous().view(-1, quad_out_list.shape[1], quad_out_list.shape[2], quad_out_list.shape[3])
        time_out_list = self.cross_attn_time(time_out_list_, quad_out_list_, quad_out_list_).contiguous().view(-1, time_out_list.shape[1], time_out_list.shape[2], time_out_list.shape[3])


        # out_pu1 = self.cross_attn_quad_poi(out_pu.transpose(0,1), quad_out_list.sum(-2), quad_out_list.sum(-2), attn_mask).transpose(0,1)
        # out_pu2 = self.cross_attn_quad_poi(out_pu.transpose(0,1), time_out_list.sum(-2), time_out_list.sum(-2), attn_mask).transpose(0,1)

        # out_pu = (out_pu1 + out_pu2) / 2
        # for i in range(len(self.quadkey_location[0])):
        #     print(self.quadkey_location[0][i].shape)
        # for i in range(len(self.quadkey_location[1])):
        #     print(self.quadkey_location[1][i].shape)
        '''
        torch.Size([2992, 106994])
        torch.Size([8548, 106994])
        torch.Size([22647, 106994])
        torch.Size([47608, 106994])

        torch.Size([2992, 8548])
        torch.Size([8548, 22647])
        torch.Size([22647, 47608])
        '''
        k = 5
        #y_linear = self.fc(out_pu)  # (seq_len, user_len, loc_count)
        dec_out_list = []
        for level, quadkey in enumerate(range(self.quadkey_num-1)):
            quad_out = self.fc1[quadkey](quad_out_list[:, :, quadkey])#.transpose(0, 1)
            dec_out_list.append(quad_out)
        #y_linear_quad = dec_out_list[-1]#.transpose(0,1)
        dec_out_list0 = []
        for level, quadkey in enumerate(range(self.quadkey_num)):
            quad_out = self.fc0[quadkey](quad_out_list2[:, :, quadkey])#.transpose(0, 1)
            dec_out_list0.append(quad_out)
        # y_linear = dec_out_list[-1]#.transpose(0,1)

        dec_time_out_list = []
        for level, timekey in enumerate(range(len(self.time_count)-1)):
            time_out = self.fc2[timekey](time_out_list[:,:,timekey])#.transpose(0,1)
            dec_time_out_list.append(time_out)

        y_linear = torch.sigmoid(self.alpha) * quad_out_list[:, :, -1] + (1-torch.sigmoid(self.alpha)) * time_out_list[:, :, -1]
        y_linear = self.fc(y_linear)
        # y_linear = dec_out_list[-1] + dec_time_out_list[-1]#.transpose(0, 1)
        # y_linear = torch.sigmoid(self.alpha) * dec_out_list[-1] + (1-torch.sigmoid(self.alpha)) * dec_time_out_list[-1]

        # y_linear = F.log_softmax(y_linear, dim=-1)

        # y_linear = self.fc1[-1](enc_out_list[-1]).transpose(0, 1)
        dec_out_list_new = []
        filter_list = []
        filter_list_0 = []
        #for level, quadkey in enumerate(range(self.quadkey_num)):
        #quad_out = self.fc1[quadkey](quad_out_list[:, :, quadkey])#.transpose(0,1)   #.permute(1,0,3,2)
        quad_out = dec_out_list[-1]
        level = 3
        quadkey = 3
        probability_distribution = torch.softmax(quad_out, dim=-1)
        # 熵的公式为 -sum(p * log(p))
        entropy = -torch.sum(probability_distribution * torch.log(probability_distribution + 1e-9), dim=-1)
        mask = entropy.unsqueeze(-1) > 0.3  #不确定的向量行置一，保留原向量行
        retain_result = torch.where(mask, torch.ones_like(mask), torch.zeros_like(mask))
        inverted_mask = ~mask  # 确定的向量行置1，保留, 跟topk搭配
        filter_result = torch.where(inverted_mask, torch.ones_like(inverted_mask), torch.zeros_like(inverted_mask))
        values, indices = torch.topk(probability_distribution, k, dim=-1)
        # 创建一个与 quad_out 相同形状的零张量
        result = torch.zeros_like(quad_out)
        result_0 = torch.zeros_like(quad_out)
        # 使用 scatter_ 将 top-k 的位置设置为 1
        result.scatter_(dim=-1, index=indices, src=values)  #这里可以尝试下给下一级加权的方式
        result_0.scatter_(dim=-1, index=indices, src=torch.ones_like(values))
        # 对每个二维切片进行稀疏矩阵乘法
        results_level = []
        results_level_0 = []
        results_location = []
        results_location_0 = []
        for i in range(result.shape[0]):
            dense_slice_t = result[i].t()
            dense_slice_t_0 = result_0[i].t()
            # if level < self.quadkey_num-1:
            #     intermediate_result_level = torch.sparse.mm(self.quadkey_location[1][level].t().to(x.device), dense_slice_t)
            #     result_level = intermediate_result_level.t()
            #     results_level.append(result_level)
            #     intermediate_result_level = torch.sparse.mm(self.quadkey_location[1][level].t().to(x.device), dense_slice_t_0)
            #     result_level_0 = intermediate_result_level.t()
            #     results_level_0.append(result_level_0)
            
            intermediate_result_location = torch.sparse.mm(self.quadkey_location[0][quadkey].t().to(x.device), dense_slice_t)
            result_location = intermediate_result_location.t()
            results_location.append(result_location)
            intermediate_result_location_0 = torch.sparse.mm(self.quadkey_location[0][quadkey].t().to(x.device), dense_slice_t_0)
            result_location_0 = intermediate_result_location_0.t()
            results_location_0.append(result_location_0)
        # if level > 0:
        #     quad_out_softmax = F.log_softmax(quad_out, dim=-1)
        #     filter_topk = torch.log(filter_list[-1] + 1e-9)
        #     quad_out = (filter_topk + quad_out_softmax) * filter_list_0[-1] * filter_result + retain_result * quad_out_softmax
        #     dec_out_list_new.append(quad_out)
        # else:
        #     dec_out_list_new.append(quad_out)
        # if level < self.quadkey_num-1:
        #     filter_list.append(torch.stack(results_level))
        #     filter_list_0.append(torch.stack(results_level_0))
        #对应的下一级topk区域
        location_topk = torch.stack(results_location, dim=0)
        location_topk_0 = torch.stack(results_location_0, dim=0)
        location_topk_softmax = torch.log(location_topk + 1e-9)
        #location_topk, location_topk_0, location_topk_softmax, retain_result, filter_result = self.project_func(x, 1, dec_out_list[1])
        y_linear_softmax = F.log_softmax(y_linear, dim=-1)
        y_linear = (y_linear_softmax + location_topk_softmax) * location_topk_0 * filter_result + y_linear_softmax * retain_result
        # location_topk, location_topk_0, location_topk_softmax, retain_result, filter_result = self.project_func(x, 3, dec_out_list[3])
        # y_linear = (y_linear + location_topk_softmax) * location_topk_0 * filter_result + y_linear * retain_result
        
        return y_linear, dec_out_list, dec_time_out_list, h, quad_out_list, dec_out_list0, location_topk, filter_result, retain_result, None  #kl_out
    # logit * filter_mask * topk + logit * retain_mask

    def project_func(self, x, quadkey, quad_out):
        dec_out_list_new = []
        filter_list = []
        filter_list_0 = []
        #for level, quadkey in enumerate(range(self.quadkey_num)):
        #quad_out = self.fc1[quadkey](quad_out_list[:, :, quadkey])#.transpose(0,1)   #.permute(1,0,3,2)
        #quad_out = dec_out_list[-1]
        # level = 3
        # quadkey = 3
        k=5
        level = quadkey
        probability_distribution = torch.softmax(quad_out, dim=-1)
        # 熵的公式为 -sum(p * log(p))
        entropy = -torch.sum(probability_distribution * torch.log(probability_distribution + 1e-9), dim=-1)
        mask = entropy.unsqueeze(-1) > 0.5  #不确定的向量行置一，保留原向量行
        retain_result = torch.where(mask, torch.ones_like(mask), torch.zeros_like(mask))
        inverted_mask = ~mask  # 确定的向量行置1，保留, 跟topk搭配
        filter_result = torch.where(inverted_mask, torch.ones_like(inverted_mask), torch.zeros_like(inverted_mask))
        values, indices = torch.topk(probability_distribution, k, dim=-1)
        # 创建一个与 quad_out 相同形状的零张量
        result = torch.zeros_like(quad_out)
        result_0 = torch.zeros_like(quad_out)
        # 使用 scatter_ 将 top-k 的位置设置为 1
        result.scatter_(dim=-1, index=indices, src=values)  #这里可以尝试下给下一级加权的方式
        result_0.scatter_(dim=-1, index=indices, src=torch.ones_like(values))
        # 对每个二维切片进行稀疏矩阵乘法
        results_level = []
        results_level_0 = []
        results_location = []
        results_location_0 = []
        for i in range(result.shape[0]):
            dense_slice_t = result[i].t()
            dense_slice_t_0 = result_0[i].t()
            # if level < self.quadkey_num-1:
            #     intermediate_result_level = torch.sparse.mm(self.quadkey_location[1][level].t().to(x.device), dense_slice_t)
            #     result_level = intermediate_result_level.t()
            #     results_level.append(result_level)
            #     intermediate_result_level = torch.sparse.mm(self.quadkey_location[1][level].t().to(x.device), dense_slice_t_0)
            #     result_level_0 = intermediate_result_level.t()
            #     results_level_0.append(result_level_0)
            
            intermediate_result_location = torch.sparse.mm(self.quadkey_location[0][quadkey].t().to(x.device), dense_slice_t)
            result_location = intermediate_result_location.t()
            results_location.append(result_location)
            intermediate_result_location_0 = torch.sparse.mm(self.quadkey_location[0][quadkey].t().to(x.device), dense_slice_t_0)
            result_location_0 = intermediate_result_location_0.t()
            results_location_0.append(result_location_0)
        # if level > 0:
        #     quad_out_softmax = F.log_softmax(quad_out, dim=-1)
        #     filter_topk = torch.log(filter_list[-1] + 1e-9)
        #     quad_out = (filter_topk + quad_out_softmax) * filter_list_0[-1] * filter_result + retain_result * quad_out_softmax
        #     dec_out_list_new.append(quad_out)
        # else:
        #     dec_out_list_new.append(quad_out)
        # if level < self.quadkey_num-1:
        #     filter_list.append(torch.stack(results_level))
        #     filter_list_0.append(torch.stack(results_level_0))
        #对应的下一级topk区域
        location_topk = torch.stack(results_location, dim=0)
        location_topk_0 = torch.stack(results_location_0, dim=0)
        location_topk_softmax = torch.log(location_topk + 1e-9)
        return location_topk, location_topk_0, location_topk_softmax, retain_result, filter_result

    def pre_enc(self, q_list):
        #expect shape: [B, N, L]
        out1_list = []
        out2_list = []
        for i in range(q_list.shape[-1] - 1):
            x_1 = q_list[..., i]
            x_2 = q_list[..., i+1] - x_1
            out1_list.append(x_1)
            out2_list.append(x_2)
        return (out1_list, out2_list)
    
    def future_multi_mixing(self, enc_out_list):
        dec_out_list = []
        for i, enc_out in zip(range(len(enc_out_list)), enc_out_list):
            dec_out = self.predict_layers[i](enc_out)
            dec_out_list.append(dec_out)

        return dec_out_list

    def out_projection(self, dec_out_origin, i, out_res):
        dec_out = self.projection_layer(dec_out_origin)
        out_res = out_res.permute(0, 2, 1)
        out_res_new = torch.zeros(self.user_len, self.seq_len, self.input_size, device=dec_out.device)
        query_res_new = torch.zeros(self.user_len, self.seq_len, self.hidden_size, device=dec_out.device)
        for j in range(1, self.seq_len+1):
            out_res = self.out_res_layers[i][j](out_res[..., :j])
            out_res_new[:, j] = self.classifier_layers[i][j](out_res).permute(0, 2, 1).sum(-1)
            query_res_new[:, j] = out_res.permute(0, 2, 1).sum(-1)
        dec_out = dec_out + out_res
        query_out = dec_out_origin + out_res_new
        return dec_out, query_out
        
    def plt_fig(self, user, q_list):
        for index in range(q_list.shape[1]):
            #for k in range(self.map_level):
            true_0 = q_list[:, index, 0].cpu().numpy().flatten().tolist()
            true_1 = q_list[:, index, 1].cpu().numpy().flatten().tolist()
            true_2 = q_list[:, index, 2].cpu().numpy().flatten().tolist()
            true_3 = q_list[:, index, 3].cpu().numpy().flatten().tolist()
            activate_user = user[0, index]
            plt.figure()
            # if k != 0:
            trend_0 = q_list[:, index, 0]
            trend_1 = q_list[:, index, 1]#.cpu().numpy().flatten().tolist()
            trend_2 = q_list[:, index, 2]
            trend_3 = q_list[:, index, 3]
            season_1 =  trend_1 - trend_0
            season_2 =  trend_2 - trend_1
            season_3 =  trend_3 - trend_2

            # season = q_list[:, index, k] - trend
            trend_0 = trend_0.cpu().numpy().flatten().tolist()
            trend_1 = trend_1.cpu().numpy().flatten().tolist()
            trend_2 = trend_2.cpu().numpy().flatten().tolist()
            trend_3 = trend_3.cpu().numpy().flatten().tolist()
            season_1 = season_1.cpu().numpy().flatten().tolist()
            season_2 = season_2.cpu().numpy().flatten().tolist()
            season_3 = season_3.cpu().numpy().flatten().tolist()
            plt.subplot(221)
            plt.plot(trend_0,color = 'red',label = 'Origin')
            plt.subplot(222)
            plt.plot(trend_3, color = 'green',label = 'Truth')
            plt.subplot(223)
            plt.plot(trend_0, color = 'orange',label = 'trend')
            plt.plot(trend_1, color = 'blue',label = 'trend')
            plt.plot(trend_2, color = 'purple',label = 'trend')
            plt.subplot(224)
            plt.plot(season_1, color = 'orange',label = 'season')
            plt.plot(season_2, color = 'blue',label = 'season')
            plt.plot(season_3, color = 'purple',label = 'season')
            # else:
            #     plt.plot(true,color = 'grey',label = 'Truth')
            plt.legend(loc="upper right")
            plt.title('Test Target')
            plt.savefig('./figure/train-{}.png'.format(activate_user))
            plt.show()




'''
~~~ h_0 strategies ~~~
Initialize RNNs hidden states
'''


def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)


class H0Strategy():

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    """ use fixed normal noise as initialization """

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        # (1, 200, 10)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)

    def on_reset(self, user):
        return self.h0


class LstmStrategy(H0Strategy):
    """ creates h0 and c0 using the inner strategy """

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return h, c

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return h, c
