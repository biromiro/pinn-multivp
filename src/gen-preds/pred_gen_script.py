# create a script baseline that should take a model name and a path to a file as input and output a prediction file
# The script should be able to run on the command line and output a prediction file

import os
import pickle
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn

total_size = 640
segment_to_skip = 100
torch.set_default_dtype(torch.float64)

sequence_length = total_size - segment_to_skip
input_channels = 5
output_channels = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleFeedForward(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SimpleFeedForward, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_channels * sequence_length, 2056)
        self.batchnorm1 = nn.BatchNorm1d(2056)
        self.fc2 = nn.Linear(2056, 1024)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 2056)
        self.batchnorm4 = nn.BatchNorm1d(2056)
        self.fc5 = nn.Linear(2056, output_channels * sequence_length)

        # Do xavier initialization
        layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)

        x = self.relu(self.batchnorm1(self.fc1(x)))
        x = self.relu(self.batchnorm2(self.fc2(x)))
        x = self.relu(self.batchnorm3(self.fc3(x)))
        x = self.relu(self.batchnorm4(self.fc4(x)))
        x = self.fc5(x)

        x = x.reshape(batch_size, self.output_channels, sequence_length)

        return x


class TensorRobustScaler:
    def __init__(self):
        self.median = None
        self.iqr = None

    def fit(self, X):
        X = X.view(-1)
        self.median = torch.quantile(X, 0.5, dim=-1)
        q1 = torch.quantile(X, 0.25, dim=-1)
        q3 = torch.quantile(X, 0.75, dim=-1)
        self.iqr = q3 - q1

    def transform(self, X):
        return (X - self.median) / self.iqr

    def inverse_transform(self, X):
        return (X * self.iqr) + self.median


def normalize(X, normalization_info):
    X_normalized = X.clone()

    for var, info in normalization_info.items():
        if info["method"] == "standardization":
            mean = info["mean"]
            std = info["std"]
            X_normalized[:, var, :] = (X_normalized[:, var, :] - mean) / std
        if info["method"] == "log_standardization":
            mean = info["mean"]
            std = info["std"]
            X_normalized[:, var, :] = (torch.log1p(
                X_normalized[:, var, :]) - mean) / std
        elif info["method"] == "log_robust_scaling":
            scaler = info["scaler"]
            X_normalized[:, var, :] = scaler.transform(
                torch.log1p(X_normalized[:, var, :]))

    return X_normalized


def denormalize(X_normalized, normalization_info):
    X_denormalized = X_normalized.clone()

    for var, info in normalization_info.items():
        if info["method"] == "standardization":
            mean = info["mean"]
            std = info["std"]
            X_denormalized[:, var, :] = (
                X_denormalized[:, var, :] * std) + mean
        if info["method"] == "log_standardization":
            mean = info["mean"]
            std = info["std"]
            X_denormalized[:, var, :] = torch.expm1(
                (X_denormalized[:, var, :] * std) + mean)
        elif info["method"] == "log_robust_scaling":
            scaler = info["scaler"]
            X_denormalized[:, var, :] = torch.expm1(
                scaler.inverse_transform(X_denormalized[:, var, :]))

    return X_denormalized


def load_model():
    model = SimpleFeedForward(input_channels, output_channels)

    model.load_state_dict(torch.load(
        'best_model_mass_momentum_conservation.pth', map_location=device))

    model.eval()
    return model


def load_data(data_path):
    X_test = torch.empty(0, total_size - segment_to_skip, 5)

    data = pd.read_csv(f'{data_path}', delimiter=',',
                       header=0, names=['R', 'L', 'n', 'v', 'T', 'B', 'A/A0', 'alpha'],
                       skiprows=0, dtype=float, na_values=['                      NaN', '                     -NaN'])

    # if data has NaN values in R column, continue to next file
    if data[['R', 'B', 'A/A0', 'alpha']].isnull().values.any():
        raise ValueError('Data has NaN values in R, B, A/A0 or alpha columns')

    X_sample = torch.tensor(
        data[['R', 'L', 'B', 'A/A0', 'alpha']].values, dtype=torch.float64)
    X_test = torch.cat(
        (X_test, X_sample[segment_to_skip:, :].unsqueeze(0)), dim=0).permute(0, 2, 1).to(torch.float64)

    X_test[:, 2, :] = torch.abs(X_test[:, 2, :])
    X_test[:, 3, :] = torch.abs(X_test[:, 3, :])

    # load normalization_info_inputs.pt
    normalization_info_inputs = torch.load(
        'normalization_info_inputs.pt', map_location=device)
    normalization_info_outputs = torch.load(
        'normalization_info_outputs.pt', map_location=device)

    return X_test, (normalization_info_inputs, normalization_info_outputs)


def make_prediction(X_test, norm_info, model):
    X_test = X_test.to(device)

    norm_inputs, norm_outputs = norm_info

    X_test_normalized = normalize(X_test, norm_inputs)

    y_pred = model(X_test_normalized[:, :, :])
    y_pred = denormalize(y_pred, norm_outputs).detach().numpy()

    # smooth predictions
    y_pred_smooth = y_pred.copy()

    for i in range(y_pred_smooth.shape[0]):
        for j in range(y_pred_smooth.shape[1]):
            y_pred_smooth[i, j, :] = pd.Series(y_pred_smooth[i, j, :]).rolling(
                window=8, min_periods=1, win_type='hamming').mean()

    return y_pred_smooth


def save_prediction(y_pred, X, folder_name, input_path):
    R = X[:, 0, :].detach().numpy()
    L = X[:, 1, :].detach().numpy()
    B = X[:, 2, :].detach().numpy()
    a_a0 = X[:, 3, :].detach().numpy()
    alpha = X[:, 4, :].detach().numpy()

    n = y_pred[:, 0, :]
    v = y_pred[:, 1, :]
    T = y_pred[:, 2, :]

    # R, L, n, v, T, B, A/A0, alpha should have 640 values in the last dimension
    # expand the last dimension to 640 by repeating the first value until index = segment_to_skip

    R = np.concatenate(
        (np.repeat(R[:, 0:1], segment_to_skip, axis=1), R), axis=1)
    L = np.concatenate(
        (np.repeat(L[:, 0:1], segment_to_skip, axis=1), L), axis=1)
    B = np.concatenate(
        (np.repeat(B[:, 0:1], segment_to_skip, axis=1), B), axis=1)
    a_a0 = np.concatenate(
        (np.repeat(a_a0[:, 0:1], segment_to_skip, axis=1), a_a0), axis=1)
    alpha = np.concatenate(
        (np.repeat(alpha[:, 0:1], segment_to_skip, axis=1), alpha), axis=1)

    n = np.concatenate(
        (np.repeat(n[:, 0:1], segment_to_skip, axis=1), n), axis=1)
    v = np.concatenate(
        (np.repeat(v[:, 0:1], segment_to_skip, axis=1), v), axis=1)
    T = np.concatenate(
        (np.repeat(T[:, 0:1], segment_to_skip, axis=1), T), axis=1)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # get file name from input_path
    file_name = os.path.basename(input_path)

    for i in range(len(B)):
        df = pd.DataFrame(np.array([R[i], L[i], n[i], v[i], T[i], B[i], a_a0[i], alpha[i]]).T, columns=[
                          'R', 'L', 'n', 'v', 'T', 'B', 'A/A0', 'alpha'])
        df.to_csv(
            f'{folder_name}/{file_name}', index=False)


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str,
                    help='Path to input.', required=True)
parser.add_argument('-o', '--output', type=str,
                    help='Path to output the file. If not provided, it will be saved in the current directory.', default='output/')

args = parser.parse_args()

model = load_model()
X_test, norm_info = load_data(args.input)
y_pred_smooth = make_prediction(X_test, norm_info, model)
save_prediction(y_pred_smooth, X_test, args.output, args.input)
