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


data = pd.read_csv(r"C:\Users\liorb\Documents\ProjectB\data_publish_v2\tang_bag2\processed\data.csv")
acce = data[['acce_x', 'acce_y', 'acce_z']].values
pos = data[['pos_x', 'pos_y', 'pos_z']].values
points = 150


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class RidiNet(nn.Module):

    def __init__(self):
        super(RidiNet, self).__init__()
        #self.first_layer = nn.Linear(points, 3) # 3xpoints to 3x3
        #self.second_layer = nn.Linear(3, 1) # 3x3 to 3x1
        #self.third_layer = nn.Linear(3, 1) # 1x3 to 1x1
        self.first_layer = nn.Sequential(nn.Linear(points*3, 500),
                                         nn.ReLU(),
                                         nn.Linear(500, 250),
                                         nn.ReLU(),
                                         nn.Linear(250, 100),
                                         nn.ReLU(),
                                         nn.Linear(100, 50),
                                         nn.ReLU(),
                                         nn.Linear(50, 1))
        #self.first_layer = nn.Linear(points*3, points)  # 3xpoints to 3x3
        #self.second_layer = nn.Linear(points, 50)  # 3x3 to 3x1
        #self.third_layer = nn.Linear(50, 1)  # 1x3 to 1x1

    def forward(self, x):
        x = torch.reshape(x, [-1])
        x = self.first_layer(x)

        #x = self.second_layer(x)

        #x = self.third_layer(x.T)

        return x


epochs = 100
learning_rate = 1e-3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
model = RidiNet()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
num_times = np.floor(np.shape(acce)[0] / points)
work_data = torch.tensor((points, 3))
distance = torch.tensor((1, 1))
output = torch.tensor((1, 1))

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    epoch_time = time.time()
    for i in range(int(np.floor(0.8*num_times))):
        work_data = torch.from_numpy(acce[points*i : points*(i+1)])
        distance = torch.norm(torch.from_numpy(pos[points * (i + 1)] - pos[points * i]))
        distance = distance.float()
        work_data = work_data.float()
        output = model(work_data)
        loss = criterion(output, distance)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
    log = "Epoch: {} | Loss: {:.4f} |".format(epoch, running_loss)
    epoch_time = time.time() - epoch_time
    log += " Epoch Time: {:.2f} secs |".format(epoch_time)
    log += " learning rate: {} ".format(get_lr(optimizer))
    print(log)
    scheduler.step()

for i in range(int(np.floor(0.2*num_times))):
    i = i + int(np.floor(0.8*num_times))
    work_data = torch.from_numpy(acce[points * i: points * (i + 1)])
    distance = torch.norm(torch.from_numpy(pos[points * (i + 1)] - pos[points * i]))
    work_data = work_data.float()
    output = model(work_data)
    log = "distance by network: {:.4f} | real distance: {:.4f} |".format(output.item(), distance)
    print(log)
'''
for i in range(10):
    i = i + 10
    work_data = torch.from_numpy(acce[points * i: points * (i + 1)])
    distance = torch.norm(torch.from_numpy(pos[points * (i + 1)] - pos[points * i]))
    work_data = work_data.float()
    output = model(work_data)
    log = "distance by network: {:.4f} | real distance: {:.4f} |".format(output.item(), distance)
    print(log)
'''
