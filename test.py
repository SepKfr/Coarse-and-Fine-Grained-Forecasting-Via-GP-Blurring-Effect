import argparse
import torch
from modules.transformer import Transformer
from data_loader import DataLoader
from train_forecast_denoise import TrainForecastDenoise


def main():
    parser = argparse.ArgumentParser(description="preprocess argument parser")
    # forecasting model
    parser.add_argument("--attn_type", type=str, default='autoformer')
    parser.add_argument("--model_name", type=str, default="autoformer")
    parser.add_argument("--exp_name", type=str, default='traffic')
    parser.add_argument("--cuda", type=str, default="cuda:0")
    parser.add_argument("--noise_type", type=str, default="gp")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--learning_residual", type=lambda x: str(x).lower() == "true", default="False")
    parser.add_argument("--no-noise", type=lambda x: str(x).lower() == "true", default="False")
    parser.add_argument("--add_noise_only_at_training", type=lambda x: str(x).lower() == "true", default="False")
    parser.add_argument("--num_epochs", type=int, default=3)

    args = parser.parse_args()

    target_col = {"traffic": "values",
                  "electricity": "power_usage",
                  "exchange": "value",
                  "solar": "Power(MW)",
                  "air_quality": "NO2",
                  "watershed": "Conductivity"
                  }

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    data_loader = DataLoader(exp_name=args.exp_name,
                             max_encoder_length=192,
                             pred_len=96,
                             target_col=target_col[args.exp_name],
                             max_train_sample=8,
                             max_test_sample=8,
                             batch_size=4,
                             device=device)

    train_enc, train_dec, train_y = next(iter(data_loader.train_loader))

    src_input_size = train_enc.shape[2]
    tgt_input_size = train_dec.shape[2]

    d_model = 32

    forecasting_model = Transformer(d_model=d_model,
                                    d_ff=d_model * 4,
                                    d_k=d_model, n_heads=8,
                                    n_layers=1, device=device,
                                    attn_type=args.attn_type,
                                    seed=1234).to(device)

    hyperparameters = {"d_model": [16, 32], "n_heads": [1, 8],
                       "n_layers": [1, 2], "lr": [0.01, 0.001]}

    trainforecastdenoise = TrainForecastDenoise(forecasting_model,
                                                data_loader.train_loader,
                                                data_loader.valid_loader,
                                                data_loader.test_loader,
                                                noise_type="gp",
                                                learning_residual=args.learning_residual,
                                                add_noise_only_at_training=args.add_noise_only_at_training,
                                                src_input_size=src_input_size,
                                                tgt_input_size=tgt_input_size,
                                                pred_len=96,
                                                hyperparameters=hyperparameters,
                                                args=args,
                                                seed=1234)

    trainforecastdenoise.train()
    trainforecastdenoise.evaluate()


if __name__ == '__main__':
    main()