# /models/embedding.py
import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]
        
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                              kernel_size=3, padding=1, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)
        
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, **_):
        super().__init__()
        self.weekday_embed = nn.Embedding(7, d_model)
        self.day_embed = nn.Embedding(32, d_model)
        self.month_embed = nn.Embedding(13, d_model)
    
    def forward(self, x):
        x = x.long()
        return self.weekday_embed(x[..., 2]) + self.day_embed(x[..., 1]) + self.month_embed(x[..., 0])
        
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, **_):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = TemporalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # Combina los tres tipos de embeddings
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
