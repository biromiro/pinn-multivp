#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
from aim import Run
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


# In[32]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[132]:


# save tensors to file
# swap last two dimensions
# X_data = X_data.permute(0, 2, 1)
# y_data = y_data.permute(0, 2, 1)
# torch.save(X_data, 'X_data.pt')
# torch.save(y_data, 'y_data.pt')


# In[33]:


# load tensors from file
X_data = torch.load('X_data.pt').to(torch.float64)
y_data = torch.load('y_data.pt').to(torch.float64)


# In[34]:


X_data.shape, y_data.shape


# In[35]:


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)


# In[36]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[37]:


# plot the y_train first channel
plt.plot(y_train[:200, 0, :].T)
plt.show()


# In[9]:


plt.loglog(y_train[:200, 0, :].T)
plt.show()


# In[10]:


# logscale the y_train and y_val first channel
y_train[:, 0, :] = torch.log(y_train[:, 0, :])
y_val[:, 0, :] = torch.log(y_val[:, 0, :])
y_test[:, 0, :] = torch.log(y_test[:, 0, :])


# In[11]:


plt.plot(y_train[:200, 0, :].T)
plt.show()


# In[12]:


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


# In[13]:


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


# create me a simple 1D convolutional network
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


# Create an instance of the SimpleConvNet
model = SimpleFeedForward(input_channels, output_channels).to(device)


# In[29]:


class CustomCriterion(nn.Module):
    def __init__(self):
        super(CustomCriterion, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.lambda_ = 2e-22

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

        nvB = ((n_pred * v_pred) / B)[:, 403:]

        # clamp nvB
        # nvB = torch.clamp(nvB, min=-1e20, max=1e20)

        # constraint_loss = torch.std(nvB, dim=1).mean().sqrt() * self.lambda_

        constant_vector_penalty = torch.sum(
            torch.abs(nvB - nvB[:, 0].unsqueeze(1)) ** 2).mean() * self.lambda_

        # Combine the two losses. You might need to tune the lambda parameter
        # to balance between the SmoothL1 loss and your constraint loss.
        total_loss = smooth_l1_loss + constant_vector_penalty
        return total_loss


# In[30]:


CustomCriterion()(X_train_normalized, y_train_normalized, y_train_normalized)


# In[20]:


# create a training loop with early stopping and checkpointing withour using pytorch-lightning


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
        x, y = x.to(device), y.to(device)
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
            x, y = x.to(device), y.to(device)
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
torch.save(best_model, 'best_model_pinn_fnn_540.pth')
