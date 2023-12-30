import torch.nn as nn
import torch
from DeepGP import DeepGPp
from modules.feedforward import PoswiseFeedForwardNet


class DenoiseModel(nn.Module):
    def __init__(self, model, gp, d_model, num_inducing, n_noise=False, iso=False):
        super(DenoiseModel, self).__init__()

        self.denoising_model = model

        self.deep_gp = DeepGPp(d_model, num_inducing)
        self.gp = gp
        self.sigma = nn.Parameter(torch.randn(1))

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.ffn_1 = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_model*4)
        self.ffn_2 = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_model*4)

        self.d = d_model
        self.n_noise = n_noise

    def add_gp_noise(self, x):

        b, s, _ = x.shape

        eps_gp, dist = self.deep_gp.predict(x)
        x_noisy = self.norm_1(x + self.ffn_1(eps_gp))

        return x_noisy, dist

    def forward(self, enc_inputs, dec_inputs):

        eps_enc = torch.randn_like(enc_inputs)
        eps_dec = torch.randn_like(dec_inputs)
        dist = None

        if self.gp:

            enc_noisy, dist_enc = self.add_gp_noise(enc_inputs)
            dec_noisy, dist = self.add_gp_noise(dec_inputs)

        elif self.n_noise:

            enc_noisy = enc_inputs
            dec_noisy = dec_inputs

        else:
            enc_noisy = enc_inputs.add_(eps_enc * torch.clip(self.sigma, 0, 0.1))
            dec_noisy = dec_inputs.add_(eps_dec * torch.clip(self.sigma, 0, 0.1))

        enc_denoise, dec_denoise = self.denoising_model(enc_noisy, dec_noisy)

        dec_output = self.norm_2(dec_inputs + self.ffn_2(dec_denoise))

        return dec_output, dist