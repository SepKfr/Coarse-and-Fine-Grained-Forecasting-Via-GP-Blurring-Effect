import torch
import torch.nn as nn
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from denoise_model import DenoiseModel


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

        super(ForecastDenoising, self).__init__()

        self.pred_len = pred_len
        self.add_noise_only_at_training = add_noise_only_at_training
        self.gp = gp
        self.lam = nn.Parameter(torch.randn(1))

        self.forecasting_model = forecasting_model

        self.de_model = DenoiseModel(self.forecasting_model,
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

        mll_error = 0
        loss = 0

        denoise = True

        if self.add_noise_only_at_training and not self.training:
            denoise = False

        enc_inputs = self.enc_embedding(enc_inputs)
        dec_inputs = self.dec_embedding(dec_inputs)

        enc_outputs, dec_outputs = self.forecasting_model(enc_inputs, dec_inputs)

        if denoise:
            de_model_outputs, dist = self.de_model(enc_outputs.clone(), dec_outputs.clone())
        else:
            de_model_outputs = dec_outputs

        final_outputs = self.final_projection(de_model_outputs[:, -self.pred_len:, :])

        if self.gp and self.training:
            mll = DeepApproximateMLL(
                VariationalELBO(self.de_model.deep_gp.likelihood, self.de_model.deep_gp, self.d_model))
            mll_error = -mll(dist, y_true).mean()

        if y_true is not None:
            mse_loss = nn.MSELoss()(y_true, final_outputs)
            loss = mse_loss + self.lam * mll_error
        return final_outputs, loss
