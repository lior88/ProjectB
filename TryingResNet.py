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

save_string = "tang_bag2"
data = pd.read_csv("C:/Users/liorb/Documents/ProjectB/data_publish_v2/" + save_string + "/processed/data.csv")
acce = data[['acce_x', 'acce_y', 'acce_z']].values
pos = data[['pos_x', 'pos_y', 'pos_z']].values

window_size = 200
num_features = 3
batch_size = 20
learning_rate = 7e-3
num_epochs = 50
allowed = 0.1  # percent of error for which we consider the result a success

save_string += '_' + str(window_size) + '_window_size'  # change according to wanted file

# print(acce.shape[0]/window_size)


acce_train, acce_test, pos_train, pos_test = train_test_split(acce, pos, test_size=0.2, random_state=5, shuffle=False)

acce_train = acce_train[1:]
acce_test = acce_test[1:]
pos_train = pos_train[1:] - pos_train[:-1]
# pos_no_diff = pos_test
pos_test = pos_test[1:] - pos_test[:-1]

train_ds = TensorDataset(torch.from_numpy(acce_train).float(), torch.from_numpy(pos_train).float())

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# loss criterion
criterion = nn.MSELoss()
train_dataloader = DataLoader(train_ds, batch_size=batch_size * window_size)

# model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)  # ResNet18
model = ResNet(BasicBlock, [3, 4, 6, 3]).to(device)  # ResNet34
# model = ResNet(BottleNeck, [3, 4, 6, 3]).to(device)  # ResNet50, not ready

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
epoch_time = time.time()

best_loss = 50
total_loss = np.zeros(num_epochs)
total_accuracy = np.zeros(num_epochs)

for epoch in range(num_epochs):
    num_correct = 0
    total_number = 0
    epoch_loss = []
    epoch_time = time.time()
    for features, targets in train_dataloader:
        distance = torch.zeros(batch_size)
        if np.shape(features)[0] != window_size * batch_size:
            continue
        features = torch.reshape(features, (batch_size, 3, window_size)).to(device)
        output = model(features)[:, 0]
        targets = torch.norm(targets, dim=1).to(device)
        for i in range(batch_size):
            distance[i] = torch.sum(targets[i * window_size:(i + 1) * window_size])
        num_correct += np.sum(((torch.abs(output - distance.to(device))) < allowed * distance.to(device)).cpu().numpy())
        total_number += batch_size
        distance = distance.to(device)
        loss = criterion(output, distance)
        optimizer.zero_grad()  # clean the gradients from previous iteration
        loss.backward()  # autograd backward to calculate gradients
        optimizer.step()  # apply update to the weights
        epoch_loss.append(loss.item())
    epoch_time = time.time() - epoch_time
    total_loss[epoch] = np.mean(epoch_loss)
    total_accuracy[epoch] = int(num_correct) * 100 / total_number
    log = 'Epoch: {} | Loss: {:.4f} |'.format(epoch, total_loss[epoch])
    log += ' Accuracy: {:.2f}% |'.format(total_accuracy[epoch])
    log += ' Epoch Time: {:.2f} secs |'.format(epoch_time)
    log += ' Learning Rate: {:.5f} |'.format(optimizer.param_groups[0]['lr'])
    print(log)
    scheduler.step()

    if total_loss[epoch] < best_loss:
        print('==> Saving model ...')
        best_loss = total_loss[epoch]
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir('our_checkpoints'):
            os.mkdir('our_checkpoints')
        torch.save(state, "./our_checkpoints/" + save_string + "_ckpt.pth")

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(np.arange(0, num_epochs), total_loss)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss vs. Epoch")
ax1.grid()

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(np.arange(0, num_epochs), total_accuracy)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy vs. Epoch")
ax2.grid()
plt.savefig("./our_checkpoints/graphs_" + save_string + ".png")
# plt.show()

state = torch.load("./our_checkpoints/" + save_string + "_ckpt.pth", map_location=device)  # there is a _best version
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
                                  [i * window_size:window_size * (i + 1)])
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
