import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import optuna
import torch.nn.functional as F
from optuna.trial import TrialState
from torch.optim import Adam
from forecast_blur_denoise import ForecastBlurDenoise


class TrainForecastBlurDenoise:
    def __init__(self,
                 *,
                 exp_name='toy_data',
                 forecasting_model_name="LSTM",
                 n_jobs=1,
                 n_trials=5,
                 num_epochs=10,
                 num_iteration,
                 forecasting_model,
                 train,
                 valid,
                 test,
                 noise_type,
                 add_noise_only_at_training,
                 input_size,
                 output_size,
                 pred_len,
                 num_inducing,
                 hyperparameters,
                 seed,
                 device):
        """
        Trainer class for training the ForecastDenoising model using Optuna hyperparameter optimization.

        Args:
        - exp_name (str): Name of the experiment (dataset).
        - forecating_model_name (str): Name of the forecasting model.
        - n_jobs (int): Total number of jobs for Optuna.
        - num_epochs (int): Total number of epochs.
        - forecasting_model (nn.Module): The underlying forecasting model.
        - train (DataLoader): DataLoader for training data.
        - valid (DataLoader): DataLoader for validation data.
        - test (DataLoader): DataLoader for test data.
        - noise_type (str): Type of noise to be added during denoising ('gp', 'iso', 'no_noise').
        - add_noise_only_at_training (bool): Flag indicating whether to add noise only during training.
        - src_input_size (int): Size of the source input.
        - tgt_input_size (int): Size of the target input.
        - tgt_output_size (int): Size of the target output.
        - pred_len (int): Length of the prediction horizon.
        - num_inducing (int): Number of inducing points for GP regression.
        - hyperparameters (dict): Hyperparameters to be optimized.
        - args: Command line arguments.
        - seed (int): Random seed for reproducibility.
        - device: Device on which to run the training.
        """

        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.num_epochs = num_epochs
        self.exp_name = exp_name
        self.forecasting_model_name = forecasting_model_name

        gp = True if noise_type == "gp" else False
        iso = True if noise_type == "iso" else False
        no_noise = True if noise_type == "no_noise" else False

        self.train_data, self.valid_data, self.test_data = train, valid, test
        self.forecasting_model = forecasting_model

        self.forecast_denoising_model = ForecastBlurDenoise(forecasting_model=forecasting_model,
                                                            gp=gp, iso=iso, no_noise=no_noise,
                                                            add_noise_only_at_training=add_noise_only_at_training,
                                                            input_size=input_size,
                                                            output_size=output_size,
                                                            pred_len=pred_len,
                                                            num_inducing=num_inducing).to(device)

        self.hyperparameters = hyperparameters
        self.best_overall_valid_loss = 1e10
        self.num_iteration = num_iteration
        self.best_forecast_denoise_model = nn.Module()
        self.model_path = "models_{}_{}".format(exp_name, pred_len)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.model_name = "{}_{}_{}_{}{}{}{}{}{}".format(forecasting_model_name, exp_name, pred_len, seed,
                                                         "_denoise",
                                                         "_gp" if gp else "",
                                                         "_predictions" if no_noise else "",
                                                         "_iso" if iso else "",
                                                         "_add_noise_only_at_training" if
                                                         add_noise_only_at_training else "")
        self.pred_len = pred_len

    def run_optuna(self):
        """
        Run the Optuna hyperparameter optimization process for the ForecastDenoising model.
        """
        study = optuna.create_study(study_name="train forecast denoise model",
                                    direction="minimize",
                                    sampler=optuna.samplers.TPESampler())
        study.optimize(self.objective, n_trials=self.n_trials,
                       n_jobs=self.n_jobs)

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
        """
        Objective function for Optuna optimization.

        Args:
        - trial: Current Optuna trial.

        Returns:
        - best_trial_valid_loss: Best validation loss achieved during the training.
        """
        param_dict = dict()
        for param, values in self.hyperparameters.items():
            param_dict[param] = trial.suggest_categorical(param, values)

        forecast_denoise_model_attributes = [attr for attr in dir(self.forecast_denoising_model) if
                                             not callable(getattr(self.forecast_denoising_model, attr))]

        for attr in forecast_denoise_model_attributes:
            if attr in self.hyperparameters.keys():
                setattr(self.forecast_denoising_model, attr, param_dict[attr])

        optimizer = Adam(self.forecast_denoising_model.parameters(), lr=param_dict["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_iteration)

        best_trial_valid_loss = 1e10

        for epoch in range(self.num_epochs):
            train_loss = 0
            self.forecast_denoising_model.train()

            for train_enc, train_dec, train_y in self.train_data:
                output_fore_den, loss = self.forecast_denoising_model(train_enc, train_dec, train_y)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

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
        """
        Train the ForecastDenoising model using Optuna hyperparameter optimization.
        """
        self.run_optuna()

    def evaluate(self):
        """
        Evaluate the performance of the best ForecastDenoising model on the test set.
        """
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

        error_path = "reported_errors_{}.csv".format(self.exp_name)

        df = pd.DataFrame.from_dict(errors, orient='index')

        if os.path.exists(error_path):
            df_old = pd.read_csv(error_path)
            df_new = pd.concat([df_old, df], axis=0)
            df_new.to_csv(error_path)
        else:
            df.to_csv(error_path)
