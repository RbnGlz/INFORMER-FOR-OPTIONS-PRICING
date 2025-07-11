# /models/decoder.py
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff, dropout, activation, **_):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self-Attention con máscara causal
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        # Cross-Attention con la salida del encoder
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        y = x = self.norm2(x)
        # Feed-forward
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.use_checkpointing = False

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            # Aplica checkpointing si está activado
            x = checkpoint(layer, x, cross, x_mask, cross_mask) if self.use_checkpointing and self.training else layer(x, cross, x_mask, cross_mask)
        
        return self.norm(x) if self.norm else x
