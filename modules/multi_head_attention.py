import random

import numpy as np
import torch
import torch.nn as nn
from forecasting_models.autoformer import AutoCorrelation


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads, attn_type, seed):

        super(MultiHeadAttention, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.WQ = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WK = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WV = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_type = attn_type
        self.seed = seed

    def forward(self, Q, K, V):

        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context, attn = AutoCorrelation(seed=self.seed)(q_s.transpose(1, 2),
                                                        k_s.transpose(1, 2),
                                                        v_s.transpose(1, 2))

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        outputs = self.fc(context)
        return outputs