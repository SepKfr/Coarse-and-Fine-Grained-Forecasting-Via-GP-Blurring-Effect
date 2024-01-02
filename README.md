# Forecastblurdenoise Package
Forecastblurdenoise is the PyTorch-based package for the research paper [Fine-grained Forecasting Models Via Gaussian Process Blurring Effect](https://arxiv.org/pdf/2312.14280.pdf). 

Methodology:
The core methodology involves training the blur model parameters end-to-end with forecasting and denoising components. This unique approach enables the underlying forecasting model to learn coarse-grained patterns, while the denoising forecaster fills in fine-grained details. The results demonstrate significant improvements over state-of-the-art models like Autoformer and Informer.

This package provides:

- The forecast-blur-denoise framework that can integrate any state-of-the-art neural time series forecasting models as the forecaster and denoiser.
- Three options for the blur model: Gaussian Process (GP), scaled isotropic noise, and no noise (perform denoising directly on predictions).
- A data loader module that works with the provided datasets in Google Drive.
- A forecasting model example (Autoformer).
- Hyperparameter tuning with Optuna.

## Datasets

In this repository, we have provided the links to google Drive of six pre-processed datasets in the following link: [Datasets](https://drive.google.com/drive/folders/1-uElnzmuCFA8aShs_O9Nlf1qyM-g90mm?usp=drive_link)

## Installation

To install simply run:

```commandline
conda install -c sepkfr forecastblurdenoise
```

## Usage Example

```python
import torch
import argparse
from forecastblurdenoise.modules.transformer import Transformer
from forecastblurdenoise.data_loader import DataLoader
from forecastblurdenoise.train_forecast_denoise import TrainForecastDenoise

# Argument parser for configuring the training process
parser = argparse.ArgumentParser(description="forecast-denoise argument parser")

# Forecasting model parameters
parser.add_argument("--attn_type", type=str, default='autoformer')
parser.add_argument("--model_name", type=str, default="autoformer")
parser.add_argument("--exp_name", type=str, default='exchange')
parser.add_argument("--cuda", type=str, default="cuda:0")
parser.add_argument("--noise_type", type=str, default="gp")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--n_trials", type=int, default=50)
parser.add_argument("--n_jobs", type=int, default=1)
parser.add_argument("--num_inducing", type=int, default=1)
parser.add_argument("--learning_residual", type=lambda x: str(x).lower() == "true", default="False")
parser.add_argument("--no-noise", type=lambda x: str(x).lower() == "true", default="False")
parser.add_argument("--add_noise_only_at_training", type=lambda x: str(x).lower() == "true", default="False")
parser.add_argument("--num_epochs", type=int, default=5)

args = parser.parse_args()

# Target column names for different datasets
target_col = {"traffic": "values",
              "electricity": "power_usage",
              "exchange": "OT",
              "solar": "Power(MW)",
              "air_quality": "NO2",
              "watershed": "Conductivity"}

# set the experiment name 
exp_name = "traffic"
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data loader configuration (replace with your own dataloader)
data_loader = DataLoader(exp_name=exp_name,
                         max_encoder_length=192,
                         pred_len=96,
                         target_col=target_col[exp_name],
                         max_train_sample=8,
                         max_test_sample=8,
                         batch_size=4,
                         device=device)

# Extracting shapes from training data for model configuration
train_enc, train_dec, train_y = next(iter(data_loader.train_loader))
src_input_size = train_enc.shape[2]
tgt_input_size = train_dec.shape[2]
tgt_output_size = train_y.shape[2]

# Dimensionality of the model
d_model = 32

# Initializing the forecasting model (replace with your forecasting model)
# setting attn_type to autoformer means we are using autocorrelation in Autoformer model
forecasting_model = Transformer(d_model=d_model,
                                d_ff=d_model * 4,
                                d_k=d_model, n_heads=8,
                                n_layers=1, device=device,
                                attn_type=args.attn_type,
                                seed=1234).to(device)

# Hyperparameter search space (change accordingly)
hyperparameters = {"d_model": [16, 32], "n_heads": [1, 8],
                   "n_layers": [1, 2], "lr": [0.01, 0.001],
                   "num_inducing": [32, 64]}

# Initializing and training the ForecastDenoise model using Optuna
trainforecastdenoise = TrainForecastDenoise(forecasting_model,
                                            data_loader.train_loader,
                                            data_loader.valid_loader,
                                            data_loader.test_loader,
                                            noise_type="gp",
                                            num_inducing=32,
                                            add_noise_only_at_training=False,
                                            src_input_size=src_input_size,
                                            tgt_input_size=tgt_input_size,
                                            tgt_output_size=tgt_output_size,
                                            pred_len=96,
                                            hyperparameters=hyperparameters,
                                            args=args,
                                            seed=1234,
                                            device=device)

# train and evaluate
trainforecastdenoise.train()
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


