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


class RidiNetShort(nn.Module):

    def __init__(self, input_dim):
        super(RidiNetShort, self).__init__()
        self.first_layer = nn.Linear(input_dim, 12)
        self.second_layer = nn.Linear(12, 6)
        self.third_layer = nn.Linear(6, 1)

    def forward(self, x):
        x = self.first_layer(x)

        x = self.second_layer(x)

        x = self.third_layer(x)

        return x


class RidiNetLong(nn.Module):

    def __init__(self, input_dim):
        super(RidiNetLong, self).__init__()
        self.first_layer = nn.Sequential(nn.Linear(input_dim*3, 500),
                                         nn.ReLU(),
                                         nn.Linear(500, 250),
                                         nn.ReLU(),
                                         nn.Linear(250, 100),
                                         nn.ReLU(),
                                         nn.Linear(100, 50),
                                         nn.ReLU(),
                                         nn.Linear(50, 1))

        #self.first_layer = nn.Linear(input_dim, 50)
        #self.second_layer = nn.Linear(100, 50)
        #self.third_layer = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.reshape(x, [-1])
        x = self.first_layer(x)

        #x = self.second_layer(x)

        #x = self.third_layer(x)

        return x


data = pd.read_csv(r"C:\Users\liorb\Documents\ProjectB\data_publish_v2\tang_bag2\processed\data.csv")
acce = data[['acce_x', 'acce_y', 'acce_z']].values
pos = data[['pos_x', 'pos_y', 'pos_z']].values
# pos_diff = pos[1:] - pos[:-1]
# print(pos_diff)
# acce = acce[1:]
# points = 100

# distance = torch.norm(torch.from_numpy(pos_diff), dim=1)
# distance = distance.float()

# print(distance)
acce_train, acce_test, pos_train, pos_test = train_test_split(acce, pos, test_size=0.2, random_state=5, shuffle=False)

acce_train = acce_train[1:]
acce_test = acce_test[1:]
pos_train = pos_train[1:] - pos_train[:-1]
pos_no_diff = pos_test
pos_test = pos_test[1:] - pos_test[:-1]

# x_scaler = StandardScaler()
# x_scaler.fit(acce_train)
# acce_train = x_scaler.transform(acce_train)
# acce_test = x_scaler.transform(acce_test)

train_ds = TensorDataset(torch.from_numpy(acce_train).float(), torch.from_numpy(pos_train).float())

num_features = 3
batch_size = 150
learning_rate = 1e-3
num_epochs = 100

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# loss criterion
criterion = nn.MSELoss()
train_dataloader = DataLoader(train_ds, batch_size=batch_size)

short = False
if short:  # input dim is 3
    # model
    model = RidiNetShort(num_features).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    epoch_time = time.time()

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_time = time.time()
        for features, targets in train_dataloader:
            # send data to device
            features = features.to(device)
            targets = torch.norm(targets, dim=1)
            targets = targets.to(device)
            # forward pass
            output = model(features)
            loss = criterion(output.view(-1), targets)
            # backward pass
            optimizer.zero_grad()  # clean the gradients from previous iteration
            loss.backward()  # autograd backward to calculate gradients
            optimizer.step()  # apply update to the weights
            epoch_losses.append(loss.item())
        epoch_time = time.time() - epoch_time
        log = 'epoch: {} loss: {:.4f}'.format(epoch, np.mean(epoch_losses))
        log += " Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)
        scheduler.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.from_numpy(acce_test).float().to(device))
        test_error = criterion(test_outputs.view(-1), torch.norm(torch.from_numpy(pos_test), dim=1).float().to(device))
    print(f'test MSE error: {test_error.item()}')

    num = int(np.floor(len(test_outputs) / batch_size))
    for i in range(num):
        check_output = test_outputs[i * batch_size:batch_size * (i + 1)]
        check_pos_new = torch.norm(torch.from_numpy(pos_test), dim=1).float()[i * batch_size:batch_size * (i + 1)]
        check_pos_old = torch.norm(
            torch.from_numpy(pos_no_diff[batch_size * (i + 1)] - pos_no_diff[batch_size * i])).float()
        print("length of output step {} is {}".format(i, torch.sum(check_output)))
        print("length of new pos step {} is {}".format(i, torch.sum(check_pos_new)))
        print("length of old pos step {} is {}".format(i, torch.sum(check_pos_old)))

elif not short:  # input dim is batch_size

    # model
    model = RidiNetLong(batch_size).to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    epoch_time = time.time()

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_time = time.time()
        for features, targets in train_dataloader:
            # send data to device
            features = features.to(device)
            if (np.shape(features)[0] != batch_size):
                continue
            targets = torch.norm(targets, dim=1)
            targets = targets.to(device)
            # forward pass
            output = model(features)
            output = torch.norm(output)
            # loss
            loss = criterion(output, torch.sum(targets))
            # backward pass
            optimizer.zero_grad()  # clean the gradients from previous iteration
            loss.backward()  # autograd backward to calculate gradients
            optimizer.step()  # apply update to the weights
            epoch_losses.append(loss.item())
        epoch_time = time.time() - epoch_time
        log = 'epoch: {} loss: {:.4f}'.format(epoch, np.mean(epoch_losses))
        log += " Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)
        scheduler.step()

    model.eval()
    num = int(np.floor(np.shape(acce_test)[0] / batch_size))
    test_error = 0
    with torch.no_grad():
        for i in range(num):
            test_outputs = model(torch.from_numpy(acce_test[i * batch_size:batch_size * (i + 1)]).float().to(device))
            test_outputs = torch.norm(test_outputs)
            check_pos_new = torch.norm(torch.from_numpy(pos_test), dim=1).float()[i * batch_size:batch_size * (i + 1)]
            check_pos_old = torch.norm(
                torch.from_numpy(pos_no_diff[batch_size * (i + 1)] - pos_no_diff[batch_size * i])).float()
            print("length of output step {} is {}".format(i, test_outputs))
            print("length of new pos step {} is {}".format(i, torch.sum(check_pos_new)))
            print("length of old pos step {} is {}".format(i, check_pos_old))
            test_error = test_error + criterion(test_outputs, torch.sum(
                torch.norm(torch.from_numpy(pos_test[i * batch_size:batch_size * (i + 1)]), dim=1).float()).to(device))
    print(f'test MSE error: {test_error.item()}')

''' 
# model 
model = RidiNet(batch_size).to(device) 
# optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

train_dataloader = DataLoader(train_ds, batch_size=batch_size) 


for epoch in range(num_epochs): 
    epoch_losses = [] 
    for features, targets in train_dataloader: 
        # send data to device 
        features = features.to(device) 
        if (np.shape(features)[0] != batch_size): 
            continue 
        targets = torch.norm(targets, dim=1) 
        targets = targets.to(device) 
        # forward pass 
        output = model(features) 
        output = torch.norm(output) 
        # loss 
        loss = criterion(output, torch.sum(targets)) 
        #loss = criterion(output.view(-1), targets) 
        # backward pass 
        optimizer.zero_grad() # clean the gradients from previous iteration 
        loss.backward() # autograd backward to calculate gradients 
        optimizer.step() # apply update to the weights 
        epoch_losses.append(loss.item()) 
    if epoch % 5 == 0: 
        print(f'epoch: {epoch} loss: {np.mean(epoch_losses)}') 

model.eval() 
num = int(np.floor(np.shape(acce_test)[0]/batch_size)) 
test_error = 0 
with torch.no_grad(): 
    for i in range(num): 
        test_outputs = model(torch.from_numpy(acce_test[i*batch_size:batch_size*(i+1)]).float().to(device)) 
        test_outputs = torch.norm(test_outputs) 
        check_pos_new = torch.norm(torch.from_numpy(pos_test), dim=1).float()[i * batch_size:batch_size * (i + 1)] 
        check_pos_old = torch.norm(torch.from_numpy(pos_later[batch_size * (i + 1)] - pos_later[batch_size * i])).float() 
        print("length of output step {} is {}".format(i, test_outputs)) 
        print("length of new pos step {} is {}".format(i, torch.sum(check_pos_new))) 
        print("length of old pos step {} is {}".format(i, check_pos_old)) 
        test_error = test_error + criterion(test_outputs, torch.sum(torch.norm(torch.from_numpy(pos_test[i*batch_size:batch_size*(i+1)]), dim=1).float()).to(device)) 
print(f'test MSE error: {test_error.item()}') 
'''
''' 
#num = int(np.floor(len(test_outputs)/batch_size)) 
for i in range(num): 
    check_output = test_outputs[i] 
    check_pos_new = torch.norm(torch.from_numpy(pos_test), dim=1).float()[i*batch_size:batch_size*(i+1)] 
    check_pos_old = torch.norm(torch.from_numpy(pos_later[batch_size * (i + 1)] - pos_later[batch_size * i])).float() 
    print("length of output step {} is {}".format(i, check_output)) 
    print("length of new pos step {} is {}".format(i, torch.sum(check_pos_new))) 
    print("length of old pos step {} is {}".format(i, torch.sum(check_pos_old))) 
'''
''' 
check_output = test_outputs[:100] 
check_pos = torch.norm(torch.from_numpy(pos_test), dim=1).float()[:100] 
print("length of output step is {}".format(torch.sum(check_output))) 
print("length of pos step is {}".format(torch.sum(check_pos))) 
'''