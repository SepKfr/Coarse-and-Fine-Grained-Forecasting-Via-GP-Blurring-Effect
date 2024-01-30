import torch.nn as nn
import torch
import random
import numpy as np
from modules.encoder import Encoder
from modules.decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, d_model,
                 d_ff, d_k, n_heads, n_layers, device, attn_type, seed):
        super(Transformer, self).__init__()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.attn_type = attn_type

        self.encoder = Encoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_k, n_heads=n_heads,
            n_layers=n_layers,
            device=device, attn_type=attn_type, seed=seed)
        self.decoder = Decoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_k, n_heads=n_heads,
            n_layers=n_layers,
            device=device,
            attn_type=attn_type, seed=seed)

    def forward(self, enc_inputs, dec_inputs):

        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_outputs)

        return enc_outputs, dec_outputs