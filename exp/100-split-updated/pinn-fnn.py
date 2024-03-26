#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from aim import Run
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import os
main_dir = '../wind-profiles/updated'
# directories = ['CR1992', 'CR2056', 'CR2071', 'CR2125', 'CR2210']
directories = []
total_size = 640
segment_to_skip = 100

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)
# three dimensional tensors, with shape (len(dir), 640, 9) and (len(dir), 640, 3)
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
        if data[['R', 'B', 'alpha', 'n', 'v', 'T']].isnull().values.any():
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
        if data[['R', 'B', 'alpha', 'n', 'v', 'T']].isnull().values.any():
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


# In[3]:


# save tensors to file
# swap last two dimensions
# X_data_train = X_data_train.permute(0, 2, 1)
# y_data_train = y_data_train.permute(0, 2, 1)
# X_data_test = X_data_test.permute(0, 2, 1)
# y_data_test = y_data_test.permute(0, 2, 1)
# torch.save(X_data_train, 'X_data_train.pt')
# torch.save(y_data_train, 'y_data_train.pt')
# torch.save(X_data_test, 'X_data_test.pt')
# torch.save(y_data_test, 'y_data_test.pt')


# In[4]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[5]:


# load tensors from file
X_train = torch.load('X_data_train.pt')
y_train = torch.load('y_data_train.pt')
X_test = torch.load('X_data_test.pt')
y_test = torch.load('y_data_test.pt')


# In[6]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[7]:


# train test split
# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)


# In[8]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[34]:


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
        tensor_robust_scaler.fit(X_var)
        normalization_info[var] = {
            "scaler": tensor_robust_scaler, "method": "robust_scaling"}

    for var in [4]:
        X_var = X_normalized[:, var, :]
        mean = X_var.mean()
        std = X_var.std()
        normalization_info[var] = {"mean": mean,
                                   "std": std, "method": "standardization"}

    return normalization_info


def normalize_inputs(X, normalization_info):
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
        elif info["method"] == "robust_scaling":
            scaler = info["scaler"]
            X_normalized[:, var, :] = scaler.transform(X_normalized[:, var, :])

    return X_normalized


def denormalize_inputs(X_normalized, normalization_info):
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
        elif info["method"] == "robust_scaling":
            scaler = info["scaler"]
            X_denormalized[:, var, :] = scaler.inverse_transform(
                X_denormalized[:, var, :])

    return X_denormalized


# In[35]:


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


def normalize_outputs(y, normalization_info):
    y_normalized = y.clone()

    for var, info in normalization_info.items():
        if info["method"] == "standardization":
            mean = info["mean"]
            std = info["std"]
            y_normalized[:, var, :] = (y_normalized[:, var, :] - mean) / std
        if info["method"] == "log_standardization":
            mean = info["mean"]
            std = info["std"]
            y_normalized[:, var, :] = (torch.log1p(
                y_normalized[:, var, :]) - mean) / std
        elif info["method"] == "robust_scaling":
            scaler = info["scaler"]
            y_normalized[:, var, :] = scaler.transform(y_normalized[:, var, :])

    return y_normalized


def denormalize_inputs(y_normalized, normalization_info):
    y_denormalized = y_normalized.clone()

    for var, info in normalization_info.items():
        if info["method"] == "standardization":
            mean = info["mean"]
            std = info["std"]
            y_denormalized[:, var, :] = (
                y_denormalized[:, var, :] * std) + mean
        if info["method"] == "log_standardization":
            mean = info["mean"]
            std = info["std"]
            y_denormalized[:, var, :] = torch.expm1(
                (y_denormalized[:, var, :] * std) + mean)
        elif info["method"] == "robust_scaling":
            scaler = info["scaler"]
            y_denormalized[:, var, :] = scaler.inverse_transform(
                y_denormalized[:, var, :])

    return y_denormalized


# In[36]:


X_normalization_info = get_normalization_info_inputs(X_train)
y_normalization_info = get_normalization_info_outputs(y_train)


# In[37]:


X_train_normalized = normalize_inputs(X_train, X_normalization_info)
X_val_normalized = normalize_inputs(X_val, X_normalization_info)
X_test_normalized = normalize_inputs(X_test, X_normalization_info)

y_train_normalized = normalize_outputs(y_train, y_normalization_info)
y_val_normalized = normalize_outputs(y_val, y_normalization_info)
y_test_normalized = normalize_outputs(y_test, y_normalization_info)


# In[38]:


# Model parameters
sequence_length = total_size - segment_to_skip
input_channels = 4
output_channels = 3
batch_size = 128


# In[41]:


# create me a simple 1D convolutional network
class SimpleFeedForward(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SimpleFeedForward, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_channels * sequence_length, 2056)
        self.batchnorm1 = nn.BatchNorm1d(2056)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(2056, 1024)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(1024, 1024)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(1024, 2056)
        self.batchnorm4 = nn.BatchNorm1d(2056)
        self.dropout4 = nn.Dropout(p=0.1)
        self.fc5 = nn.Linear(2056, output_channels * sequence_length)

        # Do xavier initialization
        layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)

        x = self.dropout1(self.relu(self.batchnorm1(self.fc1(x))))
        x = self.dropout2(self.relu(self.batchnorm2(self.fc2(x))))
        x = self.dropout3(self.relu(self.batchnorm3(self.fc3(x))))
        x = self.dropout4(self.relu(self.batchnorm4(self.fc4(x))))
        x = self.fc5(x)

        x = x.reshape(batch_size, self.output_channels, sequence_length)

        return x


# Create an instance of the SimpleConvNet
model = SimpleFeedForward(input_channels, output_channels).to(device)


# In[42]:


class CustomCriterion(nn.Module):
    def __init__(self, normalization_info_inputs, normalization_info_outputs):
        super(CustomCriterion, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.normalization_info_inputs = normalization_info_inputs
        self.normalization_info_outputs = normalization_info_outputs
        self.lambda_ = 1e-8

    def forward(self, inputs, preds, targets):
        smooth_l1_loss = self.smooth_l1_loss(preds, targets)
        inputs_denormalized = denormalize_inputs(
            inputs, self.normalization_info_inputs)
        preds_denormalized = denormalize_inputs(
            preds, self.normalization_info_outputs)

        B = inputs_denormalized[:, 2, :]
        n_pred, v_pred = preds_denormalized[:,
                                            0, :], preds_denormalized[:, 1, :]

        nvB = ((n_pred * v_pred) / B)[:, 403:]

        constraint_loss = torch.std(nvB, dim=1).mean().sqrt() * self.lambda_

        # Combine the two losses. You might need to tune the lambda parameter
        # to balance between the SmoothL1 loss and your constraint loss.
        total_loss = smooth_l1_loss + constraint_loss
        return total_loss


# In[43]:


CustomCriterion(X_normalization_info, y_normalization_info)(
    X_train_normalized, y_train_normalized, y_train_normalized)


# In[ ]:


# create a training loop with early stopping and checkpointing withour using pytorch-lightning


# Initialize a new run
run = Run(experiment="pinn-fnn")

learning_rate = 0.01
scheduler_patience = 50
scheduler_factor = 0.2
scheduler_threshold = 1e-6

# Log run parameters
run["hparams"] = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "sequence_length": sequence_length,
    "scheduler_patience": scheduler_patience,
    "scheduler_factor": scheduler_factor,
    "scheduler_threshold": scheduler_threshold,
    "loss": "SmoothL1 + Std(Nv/B) constraint"
}

# create a dataset and a dataloader
train_dataset = TensorDataset(X_train_normalized, y_train_normalized)
val_dataset = TensorDataset(X_val_normalized, y_val_normalized)
test_dataset = TensorDataset(X_test_normalized, y_test_normalized)

train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32,
                        shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32,
                         shuffle=False, num_workers=4)


# In[ ]:


# create criterion with smooth l1 loss
criterion = CustomCriterion()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# add reduce learning rate on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True, threshold=scheduler_threshold)


# In[21]:


# training loop
n_epochs = 10000
patience = 500
min_loss = np.inf
counter = 0
best_model = None

for epoch in range(n_epochs):
    print(f'Epoch {epoch}')
    model.train()
    train_loss = 0
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(x, y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    run.track({'loss': train_loss}, context={'subset': 'train'}, epoch=epoch)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            y_pred = model(x)
            loss = criterion(x, y_pred, y)
            val_loss += loss.item()
        val_loss /= len(val_loader)
        run.track({'loss': val_loss}, context={
                  'subset': 'validation'}, epoch=epoch)

    if val_loss < min_loss:
        min_loss = val_loss
        best_model = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter == patience:
            print(f'Early stopping at epoch {epoch}')
            break

    scheduler.step(train_loss)


# In[24]:


# save the best model
torch.save(best_model, 'best_model_pinn_fnn.pth')
