#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import mean_squared_error
from aim import Run

import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import os
main_dir = '../wind-profiles'
# directories = ['CR1992', 'CR2056', 'CR2071', 'CR2125', 'CR2210']
directories = []
total_size = 640
segment_to_skip = 100

# three dimensional tensors, with shape (len(dir), 640, 9) and (len(dir), 640, 3)
X_data = torch.empty(0, total_size - segment_to_skip, 4)
y_data = torch.empty(0, total_size - segment_to_skip, 3)

for directory in directories:
    for file_ in os.listdir(f'{main_dir}/{directory}'):
        data = pd.read_csv(f'{main_dir}/{directory}/{file_}', delimiter=',',
                           header=0, names=['R', 'L', 'Lon', 'Lat', 'n', 'v', 'T', 'B', 'A/A0', 'alpha', 'V/Cs', 'propag_dt'],
                           skiprows=2, dtype=float, na_values=['                      NaN', '                     -NaN'])

        # if data has NaN values in R column, continue to next file
        if data[['R', 'B', 'alpha', 'n', 'v', 'T']].isnull().values.any():
            continue

        # convert to tensor, with shape (len(data), 3), where the three channels are R, B, alpha
        X_sample = torch.tensor(
            data[['R', 'L', 'B', 'alpha']].values, dtype=torch.float32)
        y_sample = torch.tensor(
            data[['n', 'v', 'T']].values, dtype=torch.float32)

        X_data = torch.cat(
            (X_data, X_sample[segment_to_skip:, :].unsqueeze(0)), dim=0)
        y_data = torch.cat(
            (y_data, y_sample[segment_to_skip:, :].unsqueeze(0)), dim=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()

# In[3]:


# load tensors from file
X_data = torch.load('X_data.pt')
y_data = torch.load('y_data.pt')


# In[4]:


X_data.shape, y_data.shape


# In[36]:


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)


# In[37]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[20]:


# plot the y_train first channel
plt.plot(y_train[:200, 0, :].T)
plt.show()


# In[21]:


plt.loglog(y_train[:200, 0, :].T)
plt.show()


# In[38]:


# logscale the y_train and y_val first channel
y_train[:, 0, :] = torch.log(y_train[:, 0, :])
y_val[:, 0, :] = torch.log(y_val[:, 0, :])
y_test[:, 0, :] = torch.log(y_test[:, 0, :])


# In[23]:


plt.plot(y_train[:200, 0, :].T)
plt.show()


# In[24]:


# normalize data according to the maximum and minimum value in each channel
# knowing that x_train has shape (len, channels, time)

# find max and min values for each channel
max_values = X_train.max(dim=2).values.max(dim=0).values
min_values = X_train.min(dim=2).values.min(dim=0).values

diff = max_values - min_values
# if diff is zero, then the channel is constant, so we should not normalize it
diff[diff == 0] = 1

# normalize
X_train_normalized = (X_train - min_values.unsqueeze(1)) / diff.unsqueeze(1)
X_val_normalized = (X_val - min_values.unsqueeze(1)) / diff.unsqueeze(1)
X_test_normalized = (X_test - min_values.unsqueeze(1)) / diff.unsqueeze(1)


# In[25]:


# normalize y data
max_values = y_train.max(dim=2).values.max(dim=0).values
min_values = y_train.min(dim=2).values.min(dim=0).values

diff = max_values - min_values

diff[diff == 0] = 1

y_train_normalized = (y_train - min_values.unsqueeze(1)) / diff.unsqueeze(1)
y_val_normalized = (y_val - min_values.unsqueeze(1)) / diff.unsqueeze(1)
y_test_normalized = (y_test - min_values.unsqueeze(1)) / diff.unsqueeze(1)


# In[26]:


# Model parameters
sequence_length = total_size - segment_to_skip
input_channels = 4
output_channels = 3
batch_size = 128


# In[27]:


class SimpleConvLSTM(nn.Module):
    def __init__(self, input_channels, output_channels, sequence_length):
        super(SimpleConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.sequence_length = sequence_length

        # Convolutional layer to process sequence data
        self.conv1 = nn.Conv1d(in_channels=input_channels,
                               out_channels=128, kernel_size=3, padding=1)
        self.batchnorm_conv = nn.BatchNorm1d(128)

        # Bidirectional LSTM, adjusting for the output of conv layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=256,
                            num_layers=2, batch_first=True)

        # Fully connected layers
        # Adjusting for bidirectional output
        self.fc1 = nn.Linear(256 * sequence_length,
                             output_channels * sequence_length)

        # Initialization
        layers = [self.conv1, self.fc1]
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layer forward pass
        x = F.relu(self.batchnorm_conv(self.conv1(x)))

        # Adjusting dimensions for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)  # Now x is (batch, seq_len, features)

        # LSTM forward pass
        x, _ = self.lstm(x)  # No need to handle LSTM output states

        # Reshape for fully connected layers
        x = x.reshape(batch_size, -1)

        # Fully connected layers forward pass
        x = self.fc1(x)

        # Reshaping to expected output shape
        x = x.reshape(batch_size, self.output_channels, self.sequence_length)

        return x


# In[28]:
model = SimpleConvLSTM(input_channels, output_channels,
                       sequence_length).to(device)

# Initialize a new run
run = Run()

learning_rate = 0.01
scheduler_patience = 50
scheduler_factor = 0.5
scheduler_threshold = 1e-6

# Log run parameters
run["hparams"] = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "sequence_length": sequence_length,
    "scheduler_patience": scheduler_patience,
    "scheduler_factor": scheduler_factor,
    "scheduler_threshold": scheduler_threshold,
    "loss": "SmoothL1 + ConstantVectorPenalty"
}


# create criterion with smooth l1 loss
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# add reduce learning rate on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True, threshold=scheduler_threshold)


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

# training loop
n_epochs = 10000
patience = 350
min_loss = np.inf
counter = 0
best_model = None

for epoch in range(n_epochs):
    model.train()

    print(f'Epoch {epoch}')

    train_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    run.track({'loss': train_loss}, context={'subset': 'train'}, epoch=epoch)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
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

    if epoch % 100 == 0:
        torch.save(best_model, 'best_model_lstm_baseline.pth')

# In[31]:


# save the best model
torch.save(best_model, 'best_model_lstm_baseline.pth')
