import torch.nn as nn
import torch
from DeepGP import DeepGPp
from modules.feedforward import PoswiseFeedForwardNet


class BlurDenoiseModel(nn.Module):
    def __init__(self, model, d_model, num_inducing, gp, no_noise=False, iso=False):
        """
        Blur and Denoise model.

        Args:
        - model (nn.Module): Underlying forecasting model for adding and removing noise.
        - d_model (int): Dimensionality of the model.
        - num_inducing (int): Number of inducing points for the GP model.
        - gp (bool): Flag indicating whether to use GP as the blur model.
        - no_noise (bool): Flag indicating whether to add no noise during
          denoising (denoise predictions directly).
        - iso (bool): Flag indicating whether to use isotropic noise.
        """
        super(BlurDenoiseModel, self).__init__()

        self.denoising_model = model

        # Initialize DeepGP model for GP regression
        self.deep_gp = DeepGPp(d_model, num_inducing)
        self.gp = gp
        self.sigma = nn.Parameter(torch.randn(1))

        # Layer normalization and feedforward networks
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.ffn_1 = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_model*4)
        self.ffn_2 = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_model*4)

        self.d = d_model
        self.no_noise = no_noise
        self.iso = iso

    def add_gp_noise(self, x):
        """
        Add GP noise to the input using the DeepGP model.

        Args:
        - x (Tensor): Input tensor.

        Returns:
        - x_noisy (Tensor): Noisy input with added GP noise.
        - dist (Tensor): GP distribution if GP is used.
        """
        b, s, _ = x.shape

        # Predict GP noise and apply layer normalization
        eps_gp, dist = self.deep_gp.predict(x)
        x_noisy = self.norm_1(x + self.ffn_1(eps_gp))

        return x_noisy, dist

    def forward(self, enc_inputs, dec_inputs):
        """
        Forward pass of the BlurDenoiseModel.

        Args:
        - enc_inputs (Tensor): Encoder inputs.
        - dec_inputs (Tensor): Decoder inputs.

        Returns:
        - dec_output (Tensor): Denoised decoder output.
        - dist (Tensor): GP distribution if GP is used.
        """
        eps_enc = torch.randn_like(enc_inputs)
        eps_dec = torch.randn_like(dec_inputs)
        dist = None

        if self.gp:

            # Add GP noise to encoder and decoder inputs
            enc_noisy, dist_enc = self.add_gp_noise(enc_inputs)
            dec_noisy, dist = self.add_gp_noise(dec_inputs)

        elif self.iso:
            # Add scaled isotropic noise with trainable scale
            enc_noisy = enc_inputs.add_(eps_enc * torch.clip(self.sigma, 0, 0.1))
            dec_noisy = dec_inputs.add_(eps_dec * torch.clip(self.sigma, 0, 0.1))
        else:
            # No noise addition
            enc_noisy = enc_inputs
            dec_noisy = dec_inputs

        # Perform denoising with the underlying forecasting model
        enc_denoise, dec_denoise = self.denoising_model(enc_noisy, dec_noisy)

        # Apply layer normalization and feedforward network to the decoder output
        dec_output = self.norm_2(dec_inputs + self.ffn_2(dec_denoise))

        return dec_output, dist
