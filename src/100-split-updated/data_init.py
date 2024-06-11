import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import os


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


def get_normalization_info_inputs(X):
    X_normalized = X.clone()
    normalization_info = {}

    for var in [0, 1]:
        X_var = X_normalized[:, var, :]
        X_var_transformed = torch.log1p(X_var)
        mean = X_var_transformed.mean()
        std = X_var_transformed.std()
        normalization_info[var] = {"mean": mean,
                                   "std": std, "method": "log_standardization"}

    for var in [2, 3]:
        tensor_robust_scaler = TensorRobustScaler()
        X_var = X_normalized[:, var, :]
        X_var_transformed = torch.log1p(X_var)
        tensor_robust_scaler.fit(X_var_transformed)
        normalization_info[var] = {
            "scaler": tensor_robust_scaler, "method": "log_robust_scaling"}

    for var in [4]:
        X_var = X_normalized[:, var, :]
        mean = X_var.mean()
        std = X_var.std()
        normalization_info[var] = {"mean": mean,
                                   "std": std, "method": "standardization"}

    return normalization_info


def get_normalization_info_outputs(y):
    y_normalized = y.clone()
    normalization_info = {}

    for var in [0]:
        y_var = y_normalized[:, var, :]
        y_var_transformed = torch.log1p(y_var)
        mean = y_var_transformed.mean()
        std = y_var_transformed.std()
        normalization_info[var] = {"mean": mean,
                                   "std": std, "method": "log_standardization"}

    for var in [1, 2]:
        y_var = y_normalized[:, var, :]
        mean = y_var.mean()
        std = y_var.std()
        normalization_info[var] = {"mean": mean,
                                   "std": std, "method": "standardization"}

    return normalization_info


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


def get_data():
    main_dir = '../wind-profiles/updated'
    # directories = ['CR1992', 'CR2056', 'CR2071', 'CR2125', 'CR2210']
    directories = []
    total_size = 640
    segment_to_skip = 100

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=10)
    # three dimensional tensors, with shape (len(dir), 640, 9) and (len(dir), 640, 3)
    """
    X_data_train = torch.empty(0, total_size - segment_to_skip, 5)
    y_data_train = torch.empty(0, total_size - segment_to_skip, 3)
    X_data_test = torch.empty(0, total_size - segment_to_skip, 5)
    y_data_test = torch.empty(0, total_size - segment_to_skip, 3)

    for directory in directories:
        for file_ in os.listdir(f'{main_dir}/Training_files/{directory}'):
            data = pd.read_csv(f'{main_dir}/Training_files/{directory}/{file_}', delimiter=',',
                               header=0, names=['R', 'L', 'Lon', 'Lat', 'n', 'v', 'T', 'B', 'A/A0', 'alpha', 'V/Cs', 'propag_dt'],
                               skiprows=2, dtype=float, na_values=['                      NaN', '                     -NaN'])

            # if data has NaN values in R column, continue to next file
            if data[['R', 'L', 'B', 'A/A0', 'alpha', 'n', 'v', 'T']].isnull().values.any():
                continue

            # convert to tensor, with shape (len(data), 3), where the three channels are R, B, alpha
            X_sample = torch.tensor(
                data[['R', 'L', 'B', 'A/A0', 'alpha']].values, dtype=torch.float32)
            y_sample = torch.tensor(
                data[['n', 'v', 'T']].values, dtype=torch.float32)

            X_data_train = torch.cat(
                (X_data_train, X_sample[segment_to_skip:, :].unsqueeze(0)), dim=0)
            y_data_train = torch.cat(
                (y_data_train, y_sample[segment_to_skip:, :].unsqueeze(0)), dim=0)

    for directory in directories:
        for file_ in os.listdir(f'{main_dir}/Validation_files/{directory}'):
            data = pd.read_csv(f'{main_dir}/Validation_files/{directory}/{file_}', delimiter=',',
                               header=0, names=['R', 'L', 'Lon', 'Lat', 'n', 'v', 'T', 'B', 'A/A0', 'alpha', 'V/Cs', 'propag_dt'],
                               skiprows=2, dtype=float, na_values=['                      NaN', '                     -NaN'])

            # if data has NaN values in R column, continue to next file
            if data[['R', 'L', 'B', 'A/A0', 'alpha', 'n', 'v', 'T']].isnull().values.any():
                continue

            # convert to tensor, with shape (len(data), 3), where the three channels are R, B, alpha
            X_sample = torch.tensor(
                data[['R', 'L', 'B', 'A/A0', 'alpha']].values, dtype=torch.float32)
            y_sample = torch.tensor(
                data[['n', 'v', 'T']].values, dtype=torch.float32)

            X_data_test = torch.cat(
                (X_data_test, X_sample[segment_to_skip:, :].unsqueeze(0)), dim=0)
            y_data_test = torch.cat(
                (y_data_test, y_sample[segment_to_skip:, :].unsqueeze(0)), dim=0)
    """
    # load tensors from file
    X_train = torch.load('X_data_train.pt').to(torch.float64)
    y_train = torch.load('y_data_train.pt').to(torch.float64)
    X_test = torch.load('X_data_test.pt').to(torch.float64)
    y_test = torch.load('y_data_test.pt').to(torch.float64)

    X_train[:, 2, :] = torch.abs(X_train[:, 2, :])
    X_test[:, 2, :] = torch.abs(X_test[:, 2, :])
    X_train[:, 3, :] = torch.abs(X_train[:, 3, :])
    X_test[:, 3, :] = torch.abs(X_test[:, 3, :])

    # train test split
    # X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42)

    X_normalization_info = get_normalization_info_inputs(X_train)
    y_normalization_info = get_normalization_info_outputs(y_train)

    X_train_normalized = normalize(X_train, X_normalization_info)
    X_val_normalized = normalize(X_val, X_normalization_info)
    X_test_normalized = normalize(X_test, X_normalization_info)

    y_train_normalized = normalize(y_train, y_normalization_info)
    y_val_normalized = normalize(y_val, y_normalization_info)
    y_test_normalized = normalize(y_test, y_normalization_info)

    return (X_train, X_train_normalized, y_train, y_train_normalized), \
        (X_val, X_val_normalized, y_val, y_val_normalized), \
        (X_test, X_test_normalized, y_test, y_test_normalized), \
        (X_normalization_info, y_normalization_info)


def get_old_predicts(X_normalization_info, y_normalization_info):
    """
    main_dir = '../predicts_old_model'
directories = ['CR1992', 'CR2056', 'CR2071', 'CR2125', 'CR2210']
#directories = []
total_size = 640
segment_to_skip = 100

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)
# three dimensional tensors, with shape (len(dir), 640, 9) and (len(dir), 640, 3)
X_data_old_test = torch.empty(0, total_size - segment_to_skip, 5)
y_data_old_test = torch.empty(0, total_size - segment_to_skip, 3)

for directory in directories:
    for file_ in os.listdir(f'{main_dir}/{directory}'):
        data = pd.read_csv(f'{main_dir}/{directory}/{file_}', delimiter=',',
                            header=0, names=['idx', 'R', 'B', 'alpha', 'n', 'v', 'T'],
                            skiprows=0, dtype=float, na_values=['                      NaN', '                     -NaN'])

        # convert to tensor, with shape (len(data), 3), where the three channels are R, B, alpha
        X_sample = torch.tensor(
            data[['R', 'B', 'alpha']].values, dtype=torch.float32)
        y_sample = torch.tensor(
            data[['n', 'v', 'T']].values, dtype=torch.float32)

        # add 'L' and 'A/A0', with zeros, to X_sample 
        # so that it the dimensions correspond to ('R', 'L', 'B', 'A/A0', 'alpha')
        X_sample = torch.cat((X_sample, torch.zeros(X_sample.shape[0], 2)), dim=1)
        X_sample = torch.cat((X_sample, torch.zeros(X_sample.shape[0], 1)), dim=1)

        # switch the order of the columns to ('R', 'L', 'B', 'A/A0', 'alpha')
        X_sample = X_sample[:, [0, 3, 1, 4, 2]]


        X_data_old_test = torch.cat(
            (X_data_old_test, X_sample[segment_to_skip:, :].unsqueeze(0)), dim=0)
        y_data_old_test = torch.cat(
            (y_data_old_test, y_sample[segment_to_skip:, :].unsqueeze(0)), dim=0)

            # save tensors to file
# swap last two dimensions
X_data_old_test = X_data_old_test.permute(0, 2, 1)
y_data_old_test = y_data_old_test.permute(0, 2, 1)
torch.save(X_data_old_test, 'X_data_old_test.pt')
torch.save(y_data_old_test, 'y_data_old_test.pt')
    """

    # load tensors from file
    X_test = torch.load('X_data_old_test.pt').to(torch.float64)
    y_test = torch.load('y_data_old_test.pt').to(torch.float64)

    X_test[:, 2, :] = torch.abs(X_test[:, 2, :])
    X_test[:, 3, :] = torch.abs(X_test[:, 3, :])

    X_test_normalized = normalize(X_test, X_normalization_info)
    y_test_normalized = normalize(y_test, y_normalization_info)

    return X_test, X_test_normalized, y_test, y_test_normalized
