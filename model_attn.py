import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np


def clones(module, num_sub_layer):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_sub_layer)])


class FFN(nn.Module):
    def __init__(self, features, exp_factor, dropout):
        super(FFN, self).__init__()
        self.w_1 = nn.Linear(features, exp_factor * features)
        self.act = nn.ReLU()
        self.w_2 = nn.Linear(exp_factor * features, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class SelfAttn(nn.Module):
    def __init__(self, dropout):
        super(SelfAttn, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask):
        scale_term = math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale_term
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        return torch.matmul(prob, value)


class InrAttn(nn.Module):
    def __init__(self, dropout):
        super(InrAttn, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, str_mat, attn_mask):
        scale_term = math.sqrt(query.size(-1))
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale_term
        if str_mat is not None:
            str_mat = str_mat.masked_fill(attn_mask == 0.0, -1e9)
            str_mat = F.softmax(str_mat, dim=-1)
            scores += str_mat
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        return torch.matmul(prob, value)


class MHInrAttn(nn.Module):
    def __init__(self, features, n_head, dropout):
        super(MHInrAttn, self).__init__()
        self.d_h = features // n_head
        self.n_head = n_head
        self.linears = clones(nn.Linear(features, features), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, str_mat, attn_mask):
        b = x.size(0)
        query, key, value = [l(x).view(b, self.h, -1, self.d_h) for l, x in zip(self.linears, x)]
        scale_term = query.size(-1)
        str_mat = str_mat.masked_fill(attn_mask == 0.0, -1e9)
        str_mat = F.softmax(str_mat, dim=-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / scale_term + str_mat
        if attn_mask is not None:
            scores.masked_fill(attn_mask == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        x = torch.matmul(prob, value)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h*self.d_h)
        return self.linears[-1](x)

class Attn(nn.Module):
    def __init__(self, features, n_head, dropout, n_loc):
        super(Attn, self).__init__()
        self.d_h = features // n_head
        self.n_head = n_head
        self.linears = clones(nn.Linear(n_loc, features), 3)
        self.linear_map = nn.Linear(features, n_loc)
        self.dropout = nn.Dropout(dropout)
        self.h = n_head

    def forward(self, q, k, v, str_mat=None, attn_mask=None):
        b = q.size(0)
        x = [q, k, v]
        query, key, value = [l(x).view(b, -1, self.h, self.d_h).transpose(1,2) for l, x in zip(self.linears, x)]
        scale_term = query.size(-1)
        if str_mat is not None:
            str_mat = str_mat.masked_fill(attn_mask == 0.0, -1e9)
            str_mat = F.softmax(str_mat, dim=-1)
            scores = torch.matmul(query, key.transpose(-2, -1)) / scale_term + str_mat.unsqueeze(1)
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) / scale_term
        if attn_mask is not None:
            scores.masked_fill(attn_mask.unsqueeze(1).repeat(1, self.h, 1, 1) == 0.0, -1e9)
        prob = F.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        x = torch.matmul(prob, value)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.h*self.d_h)
        return self.linear_map(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, num_of_d):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.num_of_d =num_of_d

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) #+ res_att  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            #scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
            scores = scores.masked_fill(attn_mask == 0.0, -1e9)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, scores

class MultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k ,d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        batch_size, num_of_d, d_model = input_Q.shape
        batch_size, num_of_k, d_model = input_K.shape
        residual = input_Q
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, num_of_d, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, num_of_k, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, num_of_k, self.n_heads, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, scores = ScaledDotProductAttention(self.d_k, num_of_d)(Q, K, V, attn_mask)

        context = context.transpose(1, 2).reshape(batch_size, num_of_d,
                                                  self.n_heads * self.d_v).squeeze(1)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return nn.LayerNorm(self.d_model).to(self.DEVICE)(output + residual)#, res_attn