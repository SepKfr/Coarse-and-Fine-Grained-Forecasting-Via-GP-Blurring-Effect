import torch
import torch.nn as nn
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

from DeepGP import DeepGPp
from denoise_model import DenoiseModel


class ForecastDenoising(nn.Module):
    def __init__(self, forecasting_model: nn.Module,
                 gp: bool, iso: bool,
                 no_noise: bool, residual: bool,
                 add_noise_only_at_training: bool,
                 pred_len: int,
                 src_input_size: int,
                 tgt_input_size: int,
                 d_model: int = 32,
                 ):

        super(ForecastDenoising, self).__init__()

        self.pred_len = pred_len
        self.input_corrupt = add_noise_only_at_training
        self.gp = gp
        self.lam = nn.Parameter(torch.randn(1))

        self.forecasting_model = forecasting_model

        self.de_model = DenoiseModel(self.forecasting_model,
                                     gp,
                                     d_model,
                                     n_noise=no_noise,
                                     residual=residual,
                                     iso=iso)
        self.residual = residual
        self.final_projection = nn.Linear(d_model, 1)
        self.enc_embedding = nn.Linear(src_input_size, d_model)
        self.dec_embedding = nn.Linear(tgt_input_size, d_model)
        self.deep_gp = DeepGPp(d_model)

    def forward(self, enc_inputs, dec_inputs, y_true=None):

        mll_error = 0
        loss = 0
        mse_loss = 0

        enc_inputs = self.enc_embedding(enc_inputs)
        dec_inputs = self.dec_embedding(dec_inputs)

        enc_outputs, dec_outputs = self.forecasting_model(enc_inputs, dec_inputs)
        forecasting_model_outputs = self.final_projection(dec_outputs[:, -self.pred_len:, :])

        if self.input_corrupt and self.training:

            de_model_outputs, dist = self.de_model(enc_outputs.clone(), dec_outputs.clone())
            final_outputs = self.final_projection(de_model_outputs[:, -self.pred_len:, :])

            if self.gp and self.training:
                mll = DeepApproximateMLL(
                    VariationalELBO(self.de_model.deep_gp.likelihood, self.de_model.deep_gp, self.d))
                mll_error = -mll(dist, y_true.permute(2, 0, 1)).mean()

            if self.residual:

                enc_outputs_res, dec_outputs_res = self.forecasting_model(enc_outputs, dec_outputs)
                res_outputs = self.final_projection(dec_outputs_res[:, -self.pred_len:, :])
                final_outputs = forecasting_model_outputs + res_outputs
                if y_true is not None:
                    residual = y_true - forecasting_model_outputs
                    loss = nn.MSELoss()(y_true, residual)
        else:
            final_outputs = forecasting_model_outputs

        if y_true is not None:
            mse_loss = nn.MSELoss()(y_true, final_outputs)
            loss = mse_loss + self.lam * mll_error
        return final_outputs, loss
