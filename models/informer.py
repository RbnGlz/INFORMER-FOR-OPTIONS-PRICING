# /models/informer.py
import torch.nn as nn
from .embedding import DataEmbedding
from .encoder import Encoder, EncoderLayer, ConvLayer
from .decoder import Decoder, DecoderLayer
from .attention import AttentionLayer

class Informer(nn.Module):
    """
    La arquitectura completa del modelo Informer.
    Une el Encoder y el Decoder.
    """
    def __init__(self, config):
        super().__init__()
        self.pred_len = config["pred_len"]
        attn_factory = config["attention_factory"]

        self.enc_embedding = DataEmbedding(**config)
        self.dec_embedding = DataEmbedding(**config)
        
        self.encoder = Encoder(
            attn_layers=[EncoderLayer(AttentionLayer(attn_factory(mask_flag=False, **config), **config), **config) for _ in range(config["e_layers"])],
            conv_layers=[ConvLayer(config["d_model"]) for _ in range(config["e_layers"] - 1)] if config["distil"] else None,
            norm_layer=nn.LayerNorm(config["d_model"])
        )
        self.decoder = Decoder(
            layers=[DecoderLayer(
                self_attention=AttentionLayer(attn_factory(mask_flag=True, **config), **config),
                cross_attention=AttentionLayer(attn_factory(mask_flag=False, **config), **config),
                **config
            ) for _ in range(config["d_layers"])],
            norm_layer=nn.LayerNorm(config["d_model"])
        )
        
        self.encoder.use_checkpointing = self.decoder.use_checkpointing = config["use_checkpointing"]
        
        self.projection = nn.Linear(config["d_model"], config["c_out"])

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Procesa la secuencia de entrada con el encoder
        enc_out, _ = self.encoder(self.enc_embedding(x_enc, x_mark_enc))
        # Genera la secuencia de salida con el decoder
        dec_out = self.decoder(self.dec_embedding(x_dec, x_mark_dec), enc_out)
        # Proyecta la salida a la dimensión final
        return self.projection(dec_out)
