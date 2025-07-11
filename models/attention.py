# /models/attention.py
import torch
import torch.nn as nn
import numpy as np
import math
from .base import BaseAttention

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    
class XFormersAttention(BaseAttention):
    """
    Capa de atención que utiliza los kernels optimizados de xformers.
    """
    def __init__(self, mask_flag=True, **_):
        super().__init__()
        if not XFORMERS_AVAILABLE:
            raise ImportError("xformers no está instalado. Por favor, instálalo para usar XFormersAttention.")
        self.mask_flag = mask_flag

    def forward(self, queries, keys, values, attn_mask):
        # xformers aplica la máscara causal internamente
        attn_bias = xops.LowerTriangularMask() if self.mask_flag else None
        out = xops.memory_efficient_attention(queries, keys, values, attn_bias=attn_bias)
        return out, None

class ProbAttention(BaseAttention):
    """
    Implementación de ProbSparse Self-Attention del paper Informer.
    """
    def __init__(self, mask_flag=True, factor=5, attention_dropout=0.1, **_):
        super().__init__()
        self.factor = factor
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        if self.mask_flag:
            self.register_buffer("causal_mask", torch.triu(torch.ones(4096, 4096), diagonal=1).bool())

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape; _, _, L_Q, _ = Q.shape
        idx = torch.randperm(L_K, device=K.device)[:sample_k]
        K_sample = K[:, :, idx, :]
        sim = torch.einsum('bhqe,bhke->bhqk', Q, K_sample)
        metric = sim.max(dim=-1).values - sim.mean(dim=-1)
        top_idx = torch.topk(metric, n_top, dim=-1, sorted=False).indices
        Q_reduce = Q[torch.arange(B).view(B,1,1), torch.arange(H).view(1,H,1), top_idx]
        return torch.matmul(Q_reduce, K.transpose(-2, -1)), top_idx

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        return V.mean(dim=-2).unsqueeze(-2).expand(B, H, L_Q, D).clone() if not self.mask_flag else V.cumsum(dim=-2)

    def _update_context(self, context_in, V, scores, index):
        if self.mask_flag:
            scores.masked_fill_(self.causal_mask[:scores.shape[-2], :scores.shape[-1]], -float('inf'))
        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(V.shape[0])[:,None,None], torch.arange(V.shape[1])[None,:,None], index] = torch.matmul(attn, V)
        return context_in, None

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape; _, L_K, _, _ = keys.shape
        queries, keys, values = (x.transpose(1, 2) for x in (queries, keys, values))
        U, u = min(self.factor * int(np.ceil(np.log(L_K))), L_K), min(self.factor * int(np.ceil(np.log(L_Q))), L_Q)
        scores, index = self._prob_QK(queries, keys, sample_k=U, n_top=u)
        context, _ = self._update_context(self._get_initial_context(values, L_Q), values, scores * (1./math.sqrt(D)), index)
        return context.transpose(1, 2).contiguous(), None
        
class AttentionLayer(nn.Module):
    """
    Capa contenedora que proyecta Q, K, V y aplica una estrategia de atención.
    """
    def __init__(self, attention: BaseAttention, d_model, n_heads, **_):
        super().__init__()
        self.inner_attention = attention
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values, attn_mask):
        B, L, D = queries.shape; H = self.n_heads
        queries, keys, values = (p(x).view(B, -1, H, D//H) for p, x in zip((self.query_projection, self.key_projection, self.value_projection), (queries, keys, values)))
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        return self.out_projection(out.reshape(B, L, -1)), attn
