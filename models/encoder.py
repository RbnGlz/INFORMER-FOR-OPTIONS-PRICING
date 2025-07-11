# /models/encoder.py
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ConvLayer(nn.Module):
    """
    Capa de destilación para reducir la longitud de la secuencia.
    """
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        return self.pool(x).transpose(1, 2)

class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff, dropout, activation, **_):
        super().__init__()
        self.attention = attention_layer
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers, norm_layer):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers else None
        self.norm = norm_layer
        self.use_checkpointing = False

    def forward(self, x, attn_mask=None):
        for i, layer in enumerate(self.attn_layers):
            # Aplica checkpointing si está activado para ahorrar memoria
            x, _ = checkpoint(layer, x, None, None, attn_mask) if self.use_checkpointing and self.training else layer(x, attn_mask=attn_mask)
            if self.conv_layers and i < len(self.conv_layers):
                x = checkpoint(self.conv_layers[i], x) if self.use_checkpointing and self.training else self.conv_layers[i](x)
        
        return self.norm(x) if self.norm else x, None
