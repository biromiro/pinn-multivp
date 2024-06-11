#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchinfo import summary
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
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

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)
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


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# In[3]:


# load tensors from file
X_data = torch.load('X_data.pt').to(torch.float64)
y_data = torch.load('y_data.pt').to(torch.float64)


# In[4]:


X_data.shape, y_data.shape


# In[5]:


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)


# In[6]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[7]:


# plot the y_train first channel
plt.plot(y_train[:200, 0, :].T)
plt.show()


# In[8]:


plt.loglog(y_train[:200, 0, :].T)
plt.show()


# In[9]:


# logscale the y_train and y_val first channel
y_train[:, 0, :] = torch.log(y_train[:, 0, :])
y_val[:, 0, :] = torch.log(y_val[:, 0, :])
y_test[:, 0, :] = torch.log(y_test[:, 0, :])


# In[10]:


plt.plot(y_train[:200, 0, :].T)
plt.show()


# In[11]:


# normalize data according to the maximum and minimum value in each channel
# knowing that x_train has shape (len, channels, time)

# find max and min values for each channel
max_values_inputs = X_train.max(dim=2).values.max(dim=0).values
min_values_inputs = X_train.min(dim=2).values.min(dim=0).values

diff_inputs = max_values_inputs - min_values_inputs
# if diff is zero, then the channel is constant, so we should not normalize it
diff_inputs[diff_inputs == 0] = 1

# normalize
X_train_normalized = (
    X_train - min_values_inputs.unsqueeze(1)) / diff_inputs.unsqueeze(1)
X_val_normalized = (X_val - min_values_inputs.unsqueeze(1)
                    ) / diff_inputs.unsqueeze(1)
X_test_normalized = (X_test - min_values_inputs.unsqueeze(1)
                     ) / diff_inputs.unsqueeze(1)

max_values_inputs, min_values_inputs, diff_inputs


# In[12]:


# normalize y data
max_values_outputs = y_train.max(dim=2).values.max(dim=0).values
min_values_outputs = y_train.min(dim=2).values.min(dim=0).values

diff_outputs = max_values_outputs - min_values_outputs

diff_outputs[diff_outputs == 0] = 1

y_train_normalized = (
    y_train - min_values_outputs.unsqueeze(1)) / diff_outputs.unsqueeze(1)
y_val_normalized = (y_val - min_values_outputs.unsqueeze(1)
                    ) / diff_outputs.unsqueeze(1)
y_test_normalized = (y_test - min_values_outputs.unsqueeze(1)
                     ) / diff_outputs.unsqueeze(1)

max_values_outputs, min_values_outputs, diff_outputs

# In[15]:


# Model parameters
sequence_length = total_size - segment_to_skip
input_channels = 4
output_channels = 3
batch_size = 128


# In[16]:


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
                            num_layers=2, batch_first=True, bidirectional=True)

        # Fully connected layers
        # Adjusting for bidirectional output
        self.fc1 = nn.Linear(256*2 * sequence_length,
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


# Create an instance of the modified model
model = SimpleConvLSTM(input_channels, output_channels, sequence_length)


# In[17]:


class CustomCriterion(nn.Module):
    def __init__(self):
        super(CustomCriterion, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.lambda_ = 2e-8

    def forward(self, inputs, preds, targets):
        smooth_l1_loss = self.smooth_l1_loss(preds, targets)
        # Calculate SmoothL1 loss for each variable
        n_pred, v_pred = preds[:, 0, :], preds[:, 1, :]
        B = inputs[:, 2, :]

        # unnormalize the data
        n_pred = n_pred * diff_outputs[0] + min_values_outputs[0]
        v_pred = v_pred * diff_outputs[1] + min_values_outputs[1]
        B = B * diff_inputs[2] + min_values_inputs[2]

        # revert logscale on n_pred
        n_pred = torch.clamp(n_pred, min=0.1, max=25)
        n_pred = torch.exp(n_pred)

        nvB = (n_pred * v_pred) / B

        # clamp nvB
        nvB = torch.clamp(nvB, min=-1e20, max=1e20)

        constraint_loss = torch.std(nvB, dim=1).mean().sqrt() * self.lambda_

        # Combine the two losses. You might need to tune the lambda parameter
        # to balance between the SmoothL1 loss and your constraint loss.
        total_loss = smooth_l1_loss + constraint_loss
        return total_loss


# In[18]:


CustomCriterion()(X_train_normalized, y_train_normalized, y_train_normalized)


# In[19]:


# create criterion with smooth l1 loss
criterion = CustomCriterion()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# add reduce learning rate on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=50, verbose=True, threshold=1e-5)


# In[20]:


# create a training loop with early stopping and checkpointing withour using pytorch-lightning

# load tensorboard
writer = SummaryWriter('./runs/pinn-540-test2')

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


# In[21]:


# training loop
n_epochs = 10000
patience = 500
min_loss = np.inf
counter = 0
best_model = None

for epoch in range(n_epochs):
    print(f'Epoch {epoch}', end='')
    model.train()
    train_loss = 0
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(x, y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    writer.add_scalar('Loss/train', train_loss, epoch)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(x, y_pred, y)
            val_loss += loss.item()
        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)

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
    print('\033[F', end='')


# In[24]:


# save the best model
torch.save(best_model, 'best_model_pinn_lstm.pth')
