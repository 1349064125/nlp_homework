import torch
import torch.nn as nn
import math
import numpy as np
import config
from torch.autograd import Variable
import torch.nn.functional as F

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.src_emb = nn.Embedding(config.Vocab_Size, config.EmbedSize)
        self.pos_emb = PositionalEncoding(config.EmbedSize)
        self.layers = nn.ModuleList([Layer() for _ in range(config.N_Layers)])
        self.fc = nn.Linear(config.EmbedSize,2)

    def forward(self, enc_inputs,mask):

        # 1.词嵌入
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        # 2.增加位置编码
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size, src_len, d_model]
        enc_self_attn_mask = mask
        # 3.多层自注意力+残差网络
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)
        # 4.返回二分类结构
        return self.fc(enc_outputs)



class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.enc_self_attn = MutiHeadAttation(config.EmbedSize,config.N_Heads,config.Dropout)
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs

class MutiHeadAttation(nn.Module):
    def __init__(self,hid_dim,n_heads,dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim   ##词向量维度
        self.n_heads =n_heads    #头数
        self.dropout = dropout

        self.Q = nn.Linear(hid_dim,n_heads*hid_dim)
        self.K = nn.Linear(hid_dim,n_heads*hid_dim)
        self.V = nn.Linear(hid_dim,n_heads*hid_dim)

        self.sfmx = nn.Softmax(dim=-1)

        self.fc = nn.Linear(n_heads*hid_dim,hid_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self,query,key,value,mask):


        residual, batch_size = query, query.size(0)
        qlist = self.Q(query).view(batch_size,-1,self.n_heads,self.hid_dim).transpose(1,2)
        klist = self.K(key).view(batch_size,-1,self.n_heads,self.hid_dim).transpose(1,2)
        vlist = self.V(value).view(batch_size,-1,self.n_heads,self.hid_dim).transpose(1,2)

        scores= torch.matmul(qlist,klist.transpose(-1, -2)) / np.sqrt(self.hid_dim)

        mask = mask.unsqueeze(1).repeat(1, config.N_Heads, 1, 1)

        scores.masked_fill_(mask, -1e9)

        A = self.sfmx(scores)

        list = torch.matmul(A,vlist).transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.hid_dim)

        output = self.fc(list)

        return nn.LayerNorm(config.EmbedSize).to(config.DEVICE)(output+residual)






class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.EmbedSize, config.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear( config.d_ff, config.EmbedSize, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(config.EmbedSize).to(config.DEVICE)(output+residual) # [batch_size, seq_len, d_model]



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)