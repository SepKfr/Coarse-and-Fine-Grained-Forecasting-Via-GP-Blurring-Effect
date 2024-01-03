# Forecastblurdenoise Package
Forecastblurdenoise is the PyTorch-based package for the research paper [Fine-grained Forecasting Models Via Gaussian Process Blurring Effect](https://arxiv.org/pdf/2312.14280.pdf). 

Methodology:
The core methodology involves training the blur model parameters end-to-end with forecasting and denoising components. This unique approach enables the underlying forecasting model to learn coarse-grained patterns, while the denoising forecaster fills in fine-grained details. The results demonstrate significant improvements over state-of-the-art models like Autoformer and Informer.

This package provides:

- The forecast-blur-denoise framework that can integrate any state-of-the-art neural time series forecasting models as the forecaster and denoiser.
- Three options for the blur model: Gaussian Process (GP), scaled isotropic noise, and no noise (perform denoising directly on predictions).
- A data loader module that works with the provided datasets in Google Drive.
- A forecasting model example (Autoformer)
- Hyperparameter tuning with Optuna.

## Datasets

In this repository, we have provided the links to google Drive of six pre-processed datasets in the following link: [Datasets](https://drive.google.com/drive/folders/1-uElnzmuCFA8aShs_O9Nlf1qyM-g90mm?usp=drive_link)

## Installation

To install run one of the following:

```bash
pip install forecastblurdenoise==1.0.3
conda install sepkfr::forecastblurdenoise
```


## Usage Example
### Run Script for a toy dataset example

```bash
./example_usage --exp_name toy_data
```

## Command Line Args

```text
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
```

### Run as a Library 

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from forecastblurdenoise.train_forecast_blur_denoise import TrainForecastBlurDenoise

# defining a simple LSTM model

class LSTM(nn.Module):
    def __init__(self, n_layers, hidden_size):
        super(LSTM, self).__init__()

        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.n_layers = n_layers
        self.hidden_size = hidden_size

    def forward(self, input_encoder, input_decoder):

        enc_outputs, _ = self.encoder_lstm(input_encoder)
        dec_outputs, _ = self.encoder_lstm(input_decoder)
        return enc_outputs, dec_outputs

# Create a toy dataset
def create_time_series_data(num_samples,
                            input_sequence_length,
                            output_sequence_length,
                            input_size,
                            output_size,
                            device):
    # input for encoder, input for decoder, and output (ground-truth)
    return TensorDataset(torch.randn(num_samples, input_sequence_length, input_size, device=device),
                         torch.randn(num_samples, output_sequence_length, input_size, device=device),
                         torch.randn(num_samples, output_sequence_length, output_size, device=device))


# setting parameters
num_samples_train = 32
num_samples_valid = 8
num_samples_test = 8
input_sequence_length = 96
output_sequence_length = 96
batch_size = 4
input_size = 5
output_size = 1
cuda = "cuda:0"

device = torch.device(cuda if torch.cuda.is_available() else "cpu")

train_dataset = create_time_series_data(num_samples_train, input_sequence_length,
                                        output_sequence_length, input_size, output_size,
                                        device)
valid_dataset = create_time_series_data(num_samples_valid, input_sequence_length,
                                        output_sequence_length, input_size, output_size,
                                        device)
test_dataset = create_time_series_data(num_samples_test, input_sequence_length,
                                       output_sequence_length, input_size, output_size,
                                       device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

forecasting_model = LSTM(n_layers=1, hidden_size=32)

# Hyperparameter search space (change accordingly)
hyperparameters = {"d_model": [16, 32], "n_layers": [1, 2], "lr": [0.01, 0.001]}

# Initializing and training the ForecastDenoise model using Optuna
trainforecastdenoise = TrainForecastBlurDenoise(forecasting_model=forecasting_model,
                                                train=train_loader,
                                                valid=valid_loader,
                                                test=test_loader,
                                                noise_type="gp",
                                                num_inducing=32,
                                                add_noise_only_at_training=False,
                                                input_size=input_size,
                                                output_size=output_size,
                                                pred_len=96,
                                                hyperparameters=hyperparameters,
                                                seed=1234,
                                                device=device)
# train the forecast blur denoise model end-to-end
trainforecastdenoise.train()
# evaluate and save MSE, and MAE results in a csv file as reported_errors_{exp_name}.csv
trainforecastdenoise.evaluate()
```

## Citation

If you are interested in using our forecastblurdenoise model for your forcasting problem, cite our paper as:

```bibtex
@misc{koohfar2023finegrained,
      title={Fine-grained Forecasting Models Via Gaussian Process Blurring Effect}, 
      author={Sepideh Koohfar and Laura Dietz},
      year={2023},
      eprint={2312.14280},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


