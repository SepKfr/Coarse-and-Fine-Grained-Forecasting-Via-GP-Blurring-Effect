# Forecast-blur-denoise Package
Forecast-blur-denoise is the PyTorch-based package for the research paper [Fine-grained Forecasting Models Via Gaussian Process Blurring Effect](https://arxiv.org/pdf/2312.14280.pdf). This package provides:

- The forecast-blur-denoise framework that can integrate any state-of-the-art neural time series forecasting models as the forecaster and denoiser.
- Three option for the blur model: Gaussian Process (GP), scaled isotropic noise, and no noise (perform denoising directly on predictions).
- A dataloader class with the ability to work with pre-processed datasets here [Datastes](https://drive.google.com/drive/folders/1-uElnzmuCFA8aShs_O9Nlf1qyM-g90mm).
- Hyperparameter tuning with Optuna.

## Installation

To install simply run:


