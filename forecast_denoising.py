import torch
import torch.nn as nn
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from blur_denoise_model import BlurDenoiseModel


class ForecastDenoising(nn.Module):
    def __init__(self, forecasting_model: nn.Module,
                 gp: bool,
                 iso: bool,
                 no_noise: bool,
                 add_noise_only_at_training: bool,
                 pred_len: int,
                 src_input_size: int,
                 tgt_input_size: int,
                 tgt_output_size: int,
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
        - src_input_size (int): Size of the source input.
        - tgt_input_size (int): Size of the target input.
        - tgt_output_size (int): Size of the target output.
        - num_inducing (int): Number of inducing points for GP model.
        - d_model (int): Dimensionality of the model (default is 32).
        """
        super(ForecastDenoising, self).__init__()

        self.pred_len = pred_len
        self.add_noise_only_at_training = add_noise_only_at_training
        self.gp = gp
        self.lam = nn.Parameter(torch.randn(1))

        self.forecasting_model = forecasting_model

        # Initialize the blur and denoise model
        self.de_model = BlurDenoiseModel(self.forecasting_model,
                                         gp,
                                         d_model,
                                         n_noise=no_noise,
                                         iso=iso,
                                         num_inducing=num_inducing)

        self.final_projection = nn.Linear(d_model, tgt_output_size)
        self.enc_embedding = nn.Linear(src_input_size, d_model)
        self.dec_embedding = nn.Linear(tgt_input_size, d_model)
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
