import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


class CustomDataLoader:
    def __init__(self,
                 max_encoder_length,
                 pred_len,
                 max_train_sample,
                 max_test_sample,
                 batch_size,
                 device,
                 data,
                 target_col,
                 real_inputs):

        self.max_encoder_length = max_encoder_length
        self.pred_len = pred_len
        self.max_train_sample = max_train_sample * batch_size
        self.max_test_sample = max_test_sample * batch_size
        self.batch_size = batch_size

        seed = 1234

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        total_batches = int(len(data) / self.batch_size)
        train_len = int(total_batches * batch_size * 0.6)
        valid_len = int(total_batches * batch_size * 0.2)
        test_len = int(total_batches * batch_size * 0.2)

        self.real_inputs = real_inputs

        self.num_target = 1
        self.num_features = len(real_inputs) + self.num_target - 1
        self.device = device

        real_inputs.remove(target_col)

        col_names = [target_col, "id"]
        for real in real_inputs:
            col_names.append(real)

        gen_col_names = col_names[1:]
        gen_col_names.append("value")
        gen_col_names.reverse()

        data = pd.DataFrame(data[col_names].values, columns=gen_col_names)

        train_data = data[:train_len]
        valid_data = data[train_len:train_len + valid_len]
        test_data = data[train_len + valid_len:train_len + valid_len + test_len]

        self.total_time_steps = self.max_encoder_length + self.pred_len

        self.train_loader = self.create_dataloader(train_data, max_train_sample)
        self.valid_loader = self.create_dataloader(valid_data, max_test_sample)
        self.test_loader = self.create_dataloader(test_data, max_test_sample)

        train_enc, train_dec, train_y = next(iter(self.train_loader))
        self.input_size = train_enc.shape[2]
        self.output_size = train_y.shape[2]

    def create_dataloader(self, data, max_samples):

        valid_sampling_locations, split_data_map = zip(
            *[
                (
                    (identifier, self.total_time_steps + i),
                    (identifier, df)
                )
                for identifier, df in data.groupby("id")
                if (num_entries := len(df)) >= self.total_time_steps
                for i in range(num_entries - self.total_time_steps + 1)
            ]
        )
        valid_sampling_locations = list(valid_sampling_locations)
        split_data_map = dict(split_data_map)

        ranges = [
            valid_sampling_locations[i] for i in np.random.choice(
                len(valid_sampling_locations), max_samples, replace=False)
        ]
        X = torch.zeros(max_samples, self.max_encoder_length, self.num_features)
        Y = torch.zeros(max_samples, self.pred_len, 1)

        for i, tup in enumerate(ranges):

            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - self.total_time_steps: start_idx]
            val = np.array(sliced.loc[:, ~sliced.columns.isin(['id'])].values, dtype=np.float64)
            val = torch.tensor(val)
            X[i] = val[:self.max_encoder_length, :]
            Y[i] = val[-self.pred_len:, 0:1]

        dataset = TensorDataset(X[:, :-self.pred_len, :], X[:, -self.pred_len:, :], Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader
