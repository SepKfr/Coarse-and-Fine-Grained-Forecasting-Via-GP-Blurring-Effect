import os

import gpytorch
import pandas as pd
import torch
import torch.nn as nn
import optuna
import torch.nn.functional as F
from optuna.trial import TrialState
from torch.optim import Adam
from forecast_denoising import ForecastDenoising

with gpytorch.settings.num_likelihood_samples(1):
    class TrainForecastDenoise:

        def __init__(self,
                     forecasting_model,
                     train,
                     valid,
                     test,
                     noise_type,
                     learning_residual,
                     add_noise_only_at_training,
                     src_input_size,
                     tgt_input_size,
                     tgt_output_size,
                     pred_len,
                     hyperparameters,
                     args,
                     seed,
                     device,
                     num_inducing
                     ):

            gp = True if noise_type == "gp" else False
            iso = True if noise_type == "iso" else False
            no_noise = True if noise_type == "no_noise" else False

            self.train_data, self.valid_data, self.test_data = train, valid, test
            self.forecasting_model = forecasting_model

            self.forecast_denoising_model = ForecastDenoising(forecasting_model=forecasting_model,
                                                              gp=gp, iso=iso, no_noise=no_noise,
                                                              residual=learning_residual,
                                                              add_noise_only_at_training=add_noise_only_at_training,
                                                              src_input_size=src_input_size,
                                                              tgt_input_size=tgt_input_size,
                                                              tgt_output_size=tgt_output_size,
                                                              pred_len=pred_len,
                                                              num_inducing=num_inducing).to(device)

            self.hyperparameters = hyperparameters
            self.args = args
            self.best_overall_valid_loss = 1e10
            self.best_forecast_denoise_model = nn.Module()
            self.model_path = "models_{}_{}".format(args.exp_name, pred_len)
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)

            self.model_name = "{}_{}_{}_{}{}{}{}{}{}{}".format(args.model_name, args.exp_name, pred_len, seed,
                                                               "_denoise",
                                                               "_gp" if gp else "",
                                                               "_predictions" if no_noise else "",
                                                               "_iso" if iso else "",
                                                               "_learning_residual" if learning_residual else "",
                                                               "_add_noise_only_at_training" if add_noise_only_at_training else "")
            self.pred_len = pred_len

        def run_optuna(self):

            study = optuna.create_study(study_name="train forecast denoise model",
                                        direction="minimize",
                                        sampler=optuna.samplers.TPESampler())
            study.optimize(self.objective, n_trials=self.args.n_trials,
                           n_jobs=self.args.n_jobs)

            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

        def objective(self, trial):

            param_dict = dict()
            for param, values in self.hyperparameters.items():
                param_dict[param] = trial.suggest_categorical(param, values)

            forecast_denoise_model_attributes = [attr for attr in dir(self.forecast_denoising_model) if
                                                 not callable(getattr(self.forecast_denoising_model, attr))]

            for attr in forecast_denoise_model_attributes:

                if attr in self.hyperparameters.keys():

                    setattr(self.forecast_denoising_model, attr, param_dict[attr])

            optimizer = Adam(self.forecast_denoising_model.parameters(), lr=param_dict["lr"])

            best_trial_valid_loss = 1e10

            for epoch in range(self.args.num_epochs):

                train_loss = 0
                self.forecast_denoising_model.train()

                for train_enc, train_dec, train_y in self.train_data:

                    output_fore_den, loss = self.forecast_denoising_model(train_enc, train_dec, train_y)
                    train_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                trial.report(loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                self.forecast_denoising_model.eval()
                valid_loss = 0

                for valid_enc, valid_dec, valid_y in self.valid_data:

                    output, loss = self.forecast_denoising_model(valid_enc, valid_dec, valid_y)
                    valid_loss += loss.item()

                    if valid_loss < best_trial_valid_loss:
                        best_trial_valid_loss = valid_loss
                        if best_trial_valid_loss < self.best_overall_valid_loss:
                            self.best_overall_valid_loss = best_trial_valid_loss
                            self.best_forecast_denoise_model = self.forecast_denoising_model
                            torch.save({'model_state_dict': self.best_forecast_denoise_model.state_dict()},
                                       os.path.join(self.model_path, "{}".format(self.model_name)))

                return best_trial_valid_loss

        def train(self):
            self.run_optuna()

        def evaluate(self):

            self.best_forecast_denoise_model.eval()

            _, _, test_y = next(iter(self.test_data))
            total_b = len(list(iter(self.test_data)))

            predictions = torch.zeros(total_b, test_y.shape[0], self.pred_len)
            test_y_tot = torch.zeros(total_b, test_y.shape[0], self.pred_len)

            j = 0

            for test_enc, test_dec, test_y in self.test_data:
                output, _ = self.best_forecast_denoise_model(test_enc, test_dec)
                predictions[j] = output.squeeze(-1).cpu().detach()
                test_y_tot[j] = test_y[:, -self.pred_len:, :].squeeze(-1).cpu().detach()
                j += 1

            predictions = predictions.reshape(-1, 1)
            test_y = test_y_tot.reshape(-1, 1)
            normaliser = test_y.abs().mean()

            test_loss = F.mse_loss(predictions, test_y).item() / normaliser
            mse_loss = test_loss

            mae_loss = F.l1_loss(predictions, test_y).item() / normaliser
            mae_loss = mae_loss

            errors = {self.model_name: {'MSE': f"{mse_loss:.3f}", 'MAE': f"{mae_loss: .3f}"}}
            print(errors)

            error_path = "Long_horizon_Previous_set_up_Final_errors_{}.csv".format(self.args.exp_name)

            df = pd.DataFrame.from_dict(errors, orient='index')

            if os.path.exists(error_path):

                df_old = pd.read_csv(error_path)
                df_new = pd.concat([df_old, df], axis=0)
                df_new.to_csv(error_path)
            else:
                df.to_csv(error_path)