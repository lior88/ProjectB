import sys
import os
import subprocess
import argparse
import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import scipy
from scipy.spatial import distance
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Networks import ResNet, BasicBlock


data = pd.read_csv(r"C:\Users\liorb\Documents\ProjectB\data_publish_v2\tang_bag2\processed\data.csv")
acce = data[['acce_x', 'acce_y', 'acce_z']].values
pos = data[['pos_x', 'pos_y', 'pos_z']].values
print(acce.shape[0]/200)



# print(distance)
acce_train, acce_test, pos_train, pos_test = train_test_split(acce, pos, test_size=0.2, random_state=5, shuffle=False)

acce_train = acce_train[1:]
acce_test = acce_test[1:]
pos_train = pos_train[1:] - pos_train[:-1]
pos_no_diff = pos_test
pos_test = pos_test[1:] - pos_test[:-1]


train_ds = TensorDataset(torch.from_numpy(acce_train).float(), torch.from_numpy(pos_train).float())

window_size = 200
num_features = 3
batch_size = 20
learning_rate = 1e-4
num_epochs = 40

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# loss criterion
criterion = nn.MSELoss()
train_dataloader = DataLoader(train_ds, batch_size=batch_size*window_size)

#model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device) # ResNet18
model = ResNet(BasicBlock, [3, 4, 6, 8]).to(device) # ResNet34
#model = ResNet(BasicBlock, [3, 4, 6, 8]).to(device) # ResNet50

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
epoch_time = time.time()

for epoch in range(num_epochs):
    epoch_losses = []
    epoch_time = time.time()
    for features, targets in train_dataloader:
        distance = torch.zeros(batch_size)
        if np.shape(features)[0] != window_size*batch_size:
            continue
        features = torch.reshape(features, (batch_size, 3, window_size)).to(device)
        output = model(features)[:, 0]
        targets = torch.norm(targets, dim=1).to(device)
        for i in range(batch_size):
            distance[i] = torch.sum(targets[i*window_size:(i+1)*window_size])
        distance = distance.to(device)
        loss = criterion(output, distance)
        optimizer.zero_grad()  # clean the gradients from previous iteration
        loss.backward()  # autograd backward to calculate gradients
        optimizer.step()  # apply update to the weights
        epoch_losses.append(loss.item())
    epoch_time = time.time() - epoch_time
    log = 'epoch: {} loss: {:.4f}'.format(epoch, np.mean(epoch_losses))
    log += " Epoch Time: {:.2f} secs".format(epoch_time)
    print(log)
    #scheduler.step()

model.eval()
num = int(np.floor(np.shape(acce_test)[0] / window_size))
test_error = 0
with torch.no_grad():
    for i in range(num):
        test_outputs = model(torch.reshape(torch.from_numpy(acce_test[i * window_size:window_size * (i + 1)]), (1, 3, window_size)).float().to(device))
        check_pos_new = torch.norm(torch.from_numpy(pos_test), dim=1).float()[i * window_size:window_size * (i + 1)]
        check_pos_old = torch.norm(
            torch.from_numpy(pos_no_diff[window_size * (i + 1)] - pos_no_diff[window_size * i])).float()
        print("length of output step {} is {}".format(i, test_outputs.item()))
        print("length of new pos step {} is {}".format(i, torch.sum(check_pos_new)))
        print("length of old pos step {} is {}".format(i, check_pos_old))
        test_error = test_error + criterion(torch.squeeze(test_outputs), torch.sum(torch.norm(torch.from_numpy(pos_test[i * window_size:window_size * (i + 1)]), dim=1)).to(device))
print(f'test MSE error: {test_error.item()}')
