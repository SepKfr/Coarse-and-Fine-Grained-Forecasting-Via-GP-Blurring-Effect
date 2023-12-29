import random
import numpy as np
import torch
import torch.nn as nn
from modules.multi_head_attention import MultiHeadAttention
from modules.feedforward import PoswiseFeedForwardNet
from modules.encoding import PositionalEncoding


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v,
                 n_heads, attn_type, seed):

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads,
            attn_type=attn_type, seed=seed)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads,
            attn_type=attn_type, seed=seed)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, dec_inputs, enc_outputs):

        out = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs)
        out = self.layer_norm(dec_inputs + out)
        out2 = self.dec_enc_attn(out, enc_outputs, enc_outputs)
        out2 = self.layer_norm(out + out2)
        out3 = self.pos_ffn(out2)
        out3 = self.layer_norm(out2 + out3)
        return out3


class Decoder(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v,
                 n_heads, n_layers, device,
                 attn_type, seed):
        super(Decoder, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.device = device
        self.attn_type = attn_type
        self.pos_emb = PositionalEncoding(
            d_hid=d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads,
                attn_type=attn_type, seed=seed)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.d_k = d_k

    def forward(self, dec_inputs, enc_outputs):

        dec_outputs = self.pos_emb(dec_inputs)

        for layer in self.layers:
            dec_outputs = layer(
                dec_inputs=dec_outputs,
                enc_outputs=enc_outputs
            )

        return dec_outputs