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
from Networks import ResNet, BasicBlock, BottleNeck
import math

R = 6378137
f_inv = 298.257224
f = 1.0 / f_inv
e2 = (1 - f) * (1 - f)


def gps2ecef_custom(latitude, longitude, altitude):
    # (lat, lon) in WSG-84 degrees
    # altitude in meters
    cosLat = math.cos(latitude * math.pi / 180)
    sinLat = math.sin(latitude * math.pi / 180)

    cosLong = math.cos(longitude * math.pi / 180)
    sinLong = math.sin(longitude * math.pi / 180)

    c = 1 / math.sqrt(cosLat * cosLat + e2 * sinLat * sinLat)
    s = e2 * c

    x = (R*c + altitude) * cosLat * cosLong
    y = (R*c + altitude) * cosLat * sinLong
    z = (R*s + altitude) * sinLat

    return x, y, z



names = ['acce_x', 'acce_y', 'acce_z']
#data_acce = pd.read_csv(r'C:/Users\liorb\Documents\ProjectB\Processed\acce_data2_imu1.csv', names=names)
#data_acce = pd.read_csv(r'C:/Users\liorb\Documents\ProjectB\Processed\avg_method_file2.csv', names=names)
data_acce = pd.read_csv(r'C:/Users\liorb\Documents\ProjectB\Processed\voting_method_file2.csv', names=names)

#cols = ['acce_x', 'acce_z', 'acce_y']
#data_acce = data_acce[cols]
acce = data_acce.values

data_pos = pd.read_csv(r'C:/Users/liorb/Documents/ProjectB/Recordings/second/position_step2.csv')
pos = data_pos[['Height_GNSS', 'Long_GNSS', 'Lat_GNSS']].values
pos[:, 1] = pos[:, 1]/1e9
pos[:, 2] = pos[:, 2]/1e9
#pos[:, 0] = pos[:, 0]/1000


pos_array = [i for i in pos]

i = 0
for pt in pos_array:
    [xF, yF, zF] = gps2ecef_custom(pt[2], pt[1], pt[0])
    pos[i, 0] = xF
    pos[i, 1] = yF
    pos[i, 2] = zF
    i = i + 1


window_size = 200
num_features = 3
batch_size = 20
learning_rate = 7e-3
num_epochs = 50
allowed = 0.1  # percent of error for which we consider the result a success


acce_test = acce[1:]
pos_test = pos[1:] - pos[:-1]

window_size_pos = int(np.floor(window_size * (pos_test.shape[0]/acce_test.shape[0])))

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# loss criterion
criterion = nn.MSELoss()

model = ResNet(BasicBlock, [3, 4, 6, 3]).to(device)  # ResNet34


#state = torch.load("./our_checkpoints/" + "tang_bag1+2_200_window_size-for_results" + "_ckpt.pth", map_location=device)
state = torch.load("./our_checkpoints/" + "hao_handheld1_200_window_size-for_results" + "_ckpt.pth", map_location=device)  # there is a _best version
model.load_state_dict(state['net'])

model.eval()
num = int(np.floor(np.shape(acce_test)[0] / window_size))
num_test_correct = 0
test_error = 0
with torch.no_grad():
    for i in range(num):
        test_outputs = model(torch.reshape(torch.from_numpy(acce_test[i * window_size:window_size * (i + 1)]),
                                           (1, 3, window_size)).float().to(device))
        check_pos_new = torch.sum(torch.norm(torch.from_numpy(pos_test), dim=1).float()
                                  [i * window_size_pos:window_size_pos * (i + 1)])
        num_test_correct += (torch.abs(test_outputs.item() - check_pos_new)) < allowed * check_pos_new
        log = 'Step: {} | Output: {:.4f} |'.format(i, test_outputs.item())
        log += ' New position: {:.4f} |'.format(check_pos_new)
        log += ' Relative error: {:.4f}% |'.format(torch.abs(test_outputs.item() - check_pos_new) * 100 / check_pos_new)
        print(log)
        test_error += criterion(torch.squeeze(test_outputs), check_pos_new.to(device))
print('test MSE error: {:.4f} | Accuracy: {:.2f}% |'.format(test_error.item(), int(num_test_correct) * 100 / num))

# check_pos_old = torch.norm(torch.from_numpy(pos_no_diff[window_size * (i + 1)]
# - pos_no_diff[window_size * i])).float()
# log += ' old pos: {:.3f} |'.format(check_pos_old)
