import torch
import torch.nn as nn
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from DeepGP import DeepGPp


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
        self.ffn_1 = nn.Sequential(nn.Linear(d_model, d_model*4),
                                   nn.ReLU(),
                                   nn.Linear(d_model*4, d_model))
        self.ffn_2 = nn.Sequential(nn.Linear(d_model, d_model*4),
                                   nn.ReLU(),
                                   nn.Linear(d_model*4, d_model))

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


class ForecastBlurDenoise(nn.Module):
    def __init__(self, forecasting_model: nn.Module,
                 gp: bool,
                 iso: bool,
                 no_noise: bool,
                 add_noise_only_at_training: bool,
                 pred_len: int,
                 input_size: int,
                 output_size: int,
                 num_inducing: int,
                 d_model: int = 32):
        """
        Forecast-blur-denoise Module.

        Args:
        - forecasting_model (nn.Module): The underlying forecasting model.
        - gp (bool): Flag indicating whether to use GP as the blur model.
        - iso (bool): Flag indicating whether to use Gaussian isotropic for the blur model.
        - no_noise (bool): Flag indicating whether to add no noise during denoising
         (denoise predictions directly).
        - add_noise_only_at_training (bool): Flag indicating whether to add noise only during training.
        - pred_len (int): Length of the prediction horizon.
        - src_input_size (int): Number of features in input.
        - tgt_output_size (int): Number of features in output.
        - num_inducing (int): Number of inducing points for GP model.
        - d_model (int): Dimensionality of the model (default is 32).
        """
        super(ForecastBlurDenoise, self).__init__()

        self.pred_len = pred_len
        self.add_noise_only_at_training = add_noise_only_at_training
        self.gp = gp
        self.lam = nn.Parameter(torch.randn(1))

        self.forecasting_model = forecasting_model

        # Initialize the blur and denoise model
        self.de_model = BlurDenoiseModel(self.forecasting_model,
                                         d_model,
                                         gp=gp,
                                         no_noise=no_noise,
                                         iso=iso,
                                         num_inducing=num_inducing)

        self.final_projection = nn.Linear(d_model, output_size)
        self.enc_embedding = nn.Linear(input_size, d_model)
        self.dec_embedding = nn.Linear(input_size, d_model)
        self.d_model = d_model

    def forward(self, enc_inputs, dec_inputs, y_true=None):
        """
        Forward pass of the ForecastDenoising model.

        Args:
        - enc_inputs (Tensor): Encoder inputs.
        - dec_inputs (Tensor): Decoder inputs.
        - y_true (Tensor): True labels for training (optional).

        Returns:
        - final_outputs (Tensor): Model's final predictions.
        - loss (Tensor): Combined loss from denoising and forecasting components.
        """
        mll_error = 0
        loss = 0

        # Indicate whether to perform denoising
        denoise = True

        '''
        If add_noise_only_at_training flag is set and during test 
        do not perform denoising
        '''

        if self.add_noise_only_at_training and not self.training:
            denoise = False

        # Embed inputs
        enc_inputs = self.enc_embedding(enc_inputs)
        dec_inputs = self.dec_embedding(dec_inputs)

        # Get outputs from the forecasting model
        enc_outputs, dec_outputs = self.forecasting_model(enc_inputs, dec_inputs)

        if denoise:
            # Apply denoising
            de_model_outputs, dist = self.de_model(enc_outputs.clone(), dec_outputs.clone())
        else:
            de_model_outputs = dec_outputs

        # Project the denoised outputs to the final output space
        final_outputs = self.final_projection(de_model_outputs[:, -self.pred_len:, :])

        # If using GP and during training, compute MLL loss
        if self.gp and self.training:

            mll = DeepApproximateMLL(
                VariationalELBO(self.de_model.deep_gp.likelihood, self.de_model.deep_gp, self.d_model))

            mll_error = -mll(dist, y_true).mean()

        # If ground truth is available, compute MSE loss and combine with MLL loss
        if y_true is not None:
            mse_loss = nn.MSELoss()(y_true, final_outputs)
            loss = mse_loss + self.lam * mll_error

        return final_outputs, loss
