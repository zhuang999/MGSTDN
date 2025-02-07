import torch
from torch import nn
from torch.autograd import Variable
from numbers import Number
import torch.nn.functional as F
import numpy as np
import math

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        input_data = input_data.unsqueeze(1)

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden.squeeze(1)

def global_kernel(seq_len):
    mask = torch.triu(torch.ones([seq_len, seq_len]))
    matrix = torch.ones([seq_len, seq_len])
    matrix = matrix.masked_fill(mask == 0.0, -1e9)
    kernel = nn.parameter.Parameter(matrix, requires_grad=True)
    return kernel


def local_kernel(seq_len, n_session):
    mask = torch.zeros([seq_len, seq_len])
    for i in range(0, seq_len, seq_len // n_session):
        mask[i:i + seq_len // n_session, i:i + seq_len // n_session] = torch.ones(
            [seq_len // n_session, seq_len // n_session])
    mask = torch.triu(mask)
    matrix = torch.ones([seq_len, seq_len])
    matrix = matrix.masked_fill(mask == 0.0, -1e9)
    kernel = nn.parameter.Parameter(matrix, requires_grad=True)
    return kernel


class TriMixer(nn.Module):
    def __init__(self, seq_len, dims, act=nn.ReLU(), dropout=0.3):  #nn.Sigmoid()
        super().__init__()
        self.l = seq_len
        self.act = act
        self.dims = dims
        self.global_mixing = global_kernel(self.l)
        self.local_mixing = local_kernel(self.l, 2)
        self.time_encoder = TimeEncode(dims)
        self.dropout = dropout

    def forward(self, x, t):  #x [N,B,D] [B,N,D] [B,D,N]
        # t = self.time_encoder(t)
        # x = torch.cat([x, t], dim=-1)
        x1 = self.act(torch.matmul(x.permute(1, 2, 0), self.global_mixing.softmax(dim=-1)))#[:x.shape[0], :x.shape[0]]))#.permute(2, 0, 1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1 = torch.matmul(x1, self.local_mixing.softmax(dim=-1))#.permute(0, 2, 1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x = x + x1.permute(2, 0, 1)
        return x#.permute(2, 0, 1)


class TriMixer_adj(nn.Module):
    def __init__(self, seq_len, act=nn.Sigmoid()):
        super().__init__()
        self.l = seq_len
        self.act = act
        self.kernel = nn.parameter.Parameter(torch.ones([1, seq_len]), requires_grad=True)

    def forward(self, x):
        x = self.act(torch.matmul(x.permute(0, 1, 3, 2), self.kernel.softmax(dim=-1).transpose(0,1))).squeeze(-1)
        return x


"""
Module: Time-encoder
"""

class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1)))).reshape(t.shape[0], t.shape[1], -1)
        return output



"""
Module: MLP-Mixer
"""

class FeedForward(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """
    def __init__(self, input_dims, output_dims, expansion_factor, dropout=0.3):
        super().__init__()

        self.dims = output_dims
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        self.linear_0 = nn.Linear(input_dims, int(expansion_factor * self.dims))
        self.linear_1 = nn.Linear(int(expansion_factor * self.dims), self.dims)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        self.linear_1.reset_parameters()

    def forward(self, x):  #[N,B,D]
        x = self.linear_0(x)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class FeedForward_dy(nn.Module):
    """
    2-layer MLP with GeLU (fancy version of ReLU) as activation
    """
    def __init__(self, input_dims, output_dims, expansion_factor, dropout=0.3):
        super().__init__()

        self.dims = output_dims
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.linear_0 = torch.nn.Parameter(torch.Tensor(input_dims, int(expansion_factor * self.dims)))
        self.linear_1 = torch.nn.Parameter(torch.Tensor(int(expansion_factor * self.dims), self.dims))
        stdv = 1. / math.sqrt(self.linear_0.shape[0])
        self.linear_0.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.linear_1.shape[0])
        self.linear_1.data.uniform_(-stdv, stdv)

        #self.reset_parameters()

    def reset_parameters(self):
        self.linear_0.reset_parameters()
        self.linear_1.reset_parameters()

    def forward(self, x):  #[N,B,D]
        seq_len = x.shape[-1]
        x = torch.matmul(x, self.linear_0[:seq_len].unsqueeze(0))
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = torch.matmul(x, self.linear_1[:, :seq_len].unsqueeze(0))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class MixerBlock(nn.Module):
    """
    out = X.T + MLP_Layernorm(X.T)     # apply token mixing
    out = out.T + MLP_Layernorm(out.T) # apply channel mixing
    """
    def __init__(self, dims, 
                 token_expansion_factor=2,
                 dropout=0.5):
        super().__init__()
        self.time_encoder = TimeEncode(dims)
        self.token_layernorm = nn.LayerNorm(dims)
        self.token_forward = FeedForward(dims, dims, token_expansion_factor, dropout)
        # self.feature_layernorm = nn.LayerNorm(sequence_length)
        # self.feature_forward = FeedForward_dy(sequence_length, sequence_length, token_expansion_factor, dropout)
        # self.alpha = nn.Parameter(torch.FloatTensor(dims))
        # stdv = 1. / math.sqrt(self.alpha.shape[0])
        # self.alpha.data.uniform_(-stdv, stdv)
        # self.sequence_length = sequence_length

    def reset_parameters(self):
        self.token_layernorm.reset_parameters()
        self.token_forward.reset_parameters()
        
    def token_mixer(self, feat, t):  #[N,B,D]
        seq_len, bz, dim = feat.shape
        # t = self.time_encoder(t)
        # x = torch.cat([x, t], dim=-1)
        x1 = self.token_layernorm(feat)#.permute(0, 2, 1)
        x1 = self.token_forward(x1)#.permute(0, 2, 1)
        #x1 += feat
        #feat = feat + x1
        # x2 = feat.permute(1,2,0)
        # x2_list = []
        # for i in range(1,seq_len+1):
        #     x2_single = x2[:, :, :i]
        #     x2_single = nn.LayerNorm(i).to(feat.device)(x2_single)
        #     #x2_single = nn.functional.pad(x2_single, pad=(0,self.sequence_length-i,0,0))
        #     x2_single = self.feature_forward(x2_single).sum(-1)
        #     x2_list.append(x2_single)
        # x2 = torch.stack(x2_list, dim=0)
        # feat = feat + x2 
        # #x = torch.sigmoid(self.alpha) * x1 + (1 - torch.sigmoid(self.alpha)) * x2
        return x1
    

    def forward(self, x, t):
        x = x + self.token_mixer(x, t)
        return x

class FeatEncode(nn.Module):
    """
    Return [raw_edge_feat | TimeEncode(edge_time_stamp)]
    """
    def __init__(self, time_dims, feat_dims, out_dims):
        super().__init__()
        
        self.time_encoder = TimeEncode(time_dims)
        self.feat_encoder = nn.Linear(time_dims + feat_dims, out_dims) 
        self.reset_parameters()

    def reset_parameters(self):
        self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()
        
    def forward(self, edge_feats, edge_ts):
        edge_time_feats = self.time_encoder(edge_ts)
        x = torch.cat([edge_feats, edge_time_feats], dim=1)
        return self.feat_encoder(x)

class MLPMixer(nn.Module):
    """
    Input : [ batch_size, graph_size, edge_dims+time_dims]
    Output: [ batch_size, graph_size, output_dims]
    """
    def __init__(self, per_graph_size, time_channels,
                 input_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5,
                 token_expansion_factor=0.5, 
                 channel_expansion_factor=4, 
                 module_spec=None, use_single_layer=False
                ):
        super().__init__()
        self.per_graph_size = per_graph_size

        self.num_layers = num_layers
        
        # input & output classifer
        self.feat_encoder = FeatEncode(time_channels, input_channels, hidden_channels)
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.mlp_head = nn.Linear(hidden_channels, out_channels)
        
        # inner layers
        self.mixer_blocks = torch.nn.ModuleList()
        for ell in range(num_layers):
            if module_spec is None:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, hidden_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=None, 
                               use_single_layer=use_single_layer)
                )
            else:
                self.mixer_blocks.append(
                    MixerBlock(per_graph_size, hidden_channels, 
                               token_expansion_factor, 
                               channel_expansion_factor, 
                               dropout, module_spec=module_spec[ell], 
                               use_single_layer=use_single_layer)
                )



        # init
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mixer_blocks:
            layer.reset_parameters()
        self.feat_encoder.reset_parameters()
        self.layernorm.reset_parameters()
        self.mlp_head.reset_parameters()

    def forward(self, edge_feats, edge_ts, batch_size, inds):
        # x :     [ batch_size, graph_size, edge_dims+time_dims]
        edge_time_feats = self.feat_encoder(edge_feats, edge_ts)
        x = torch.zeros((batch_size * self.per_graph_size, 
                         edge_time_feats.size(1))).to(edge_feats.device)
        x[inds] = x[inds] + edge_time_feats     
        x = torch.split(x, self.per_graph_size)
        x = torch.stack(x)
        
        # apply to original feats
        for i in range(self.num_layers):
            # apply to channel + feat dim
            x = self.mixer_blocks[i](x)
        x = self.layernorm(x)
        x = torch.mean(x, dim=1)
        x = self.mlp_head(x)
        return x

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio=0.5):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        self.token_layernorm = nn.LayerNorm(input_dim)
        for _ in range(num_layers):
            self.layers.append(FeedForward(input_dim, output_dim, hidden_dim/input_dim, dropout_ratio))
        # if num_layers == 1:
        #     self.layers.append(nn.Linear(input_dim, output_dim))
        # else:
        #     self.layers.append(nn.Linear(input_dim, hidden_dim))
        #     for i in range(num_layers - 2):
        #         self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        #     self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        
        h_list = []
        for l, layer in enumerate(self.layers):
            h = feats.clone()
            h = self.token_layernorm(h)
            h = layer(h)
            feats += h
            h_list.append(feats)
        h_list = torch.stack(h_list, dim=0)
        feats = torch.sum(h_list, dim=0)

        return h_list, feats

class LinearBlock(nn.Module):
 def __init__(self, dim, mlp_ratio=4, drop=0.5, act_layer=nn.GELU,
        norm=nn.LayerNorm): 
        super().__init__()
        # FF over features
        self.mlp1 = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm1 = norm(dim)
        # FF over patches
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm(dim)
 def forward(self, x):
        x = x + self.mlp1(self.norm1(x))
        #x = x + self.mlp2(self.norm2(x))
        return x
    
class Mlp(nn.Module):
 def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
 def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x