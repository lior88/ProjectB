import sys
import os
import subprocess
import argparse
import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import scipy
from scipy.spatial import distance
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Networks import ResNet, BasicBlock, BottleNeck, SimpleNet
from Networks import calc_conv_dim
import random
from ray import tune, init, shutdown
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
matplotlib.use('pdf')

torch.manual_seed(30)
np.random.seed(30)
random.seed(30)

MODEL_WEIGHTS_PATH = os.path.join(os.getcwd(), "best_model", "parameters.pt")


def train_data(config, train_ds, save_string, abs_value=False, window_size=50, num_epochs=50, data_dir=None,
               checkpoint_dir=None):
    # model
    # model = ResNet(BasicBlock, [2, 2, 2, 2], abs_flag=abs_value).to(device)  # ResNet18
    # model = ResNet(BasicBlock, [3, 4, 6, 3], abs_flag=abs_value).to(device)  # ResNet34
    # model = ResNet(BottleNeck, [3, 4, 6, 3]).to(device)  # ResNet50, not ready
    # model = SimpleNet(config["out1"], config["out2"], config["out3"], abs_flag=abs_value)
    model = SimpleNet(window_size, config["out1"], config["out2"], abs_flag=abs_value)
    # device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    device = torch.device("cpu")

    # loss criterion
    criterion = nn.MSELoss()
    # train and validation
    train_set_size = int(len(train_ds) * 0.8)
    train_subset, val_subset = random_split(train_ds, [train_set_size, len(train_ds) - train_set_size])
    train_loader = DataLoader(train_subset, batch_size=int(config["batch_size"] * window_size), shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=int(config["batch_size"] * window_size), shuffle=False)
    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    epoch_time = time.time()

    best_loss = 50
    total_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        # training loss
        total_number = 0
        epoch_loss = []
        epoch_time = time.time()
        for features, targets in train_loader:
            distance = torch.zeros(config["batch_size"])
            if np.shape(features)[0] != window_size * config["batch_size"]:
                continue
            if abs_value:
                in_channel = 1
            else:
                in_channel = 3
            features = torch.reshape(features, (config["batch_size"], in_channel, window_size)).to(device)
            output = model(features)[:, 0]
            targets = torch.norm(targets, dim=1).to(device)
            for i in range(config["batch_size"]):
                distance[i] = torch.sum(targets[i * window_size:(i + 1) * window_size])
                if total_number == 5 * config["batch_size"]:
                    res = ' Real Value: {:.4f} | Predicted Value: {:.4f} |'.format(distance[i], output[i])
                    print(res)
            total_number += config["batch_size"]
            distance = distance.to(device)
            loss = criterion(output, distance)
            optimizer.zero_grad()  # clean the gradients from previous iteration
            loss.backward()  # autograd backward to calculate gradients
            optimizer.step()  # apply update to the weights
            epoch_loss.append(loss.item())
        epoch_time = time.time() - epoch_time
        total_loss[epoch] = np.mean(epoch_loss)
        log = 'Epoch: {} | Loss: {:.4f} |'.format(epoch, total_loss[epoch])
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
            '''
            if not os.path.isdir('our_checkpoints'):
                os.mkdir('our_checkpoints')
            torch.save(state, "./our_checkpoints/" + save_string + "_ckpt.pth")
            '''

        # validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        for features, targets in val_loader:
            with torch.no_grad():
                distance = torch.zeros(config["batch_size"])
                if np.shape(features)[0] != window_size * config["batch_size"]:
                    continue
                if abs_value:
                    in_channel = 1
                else:
                    in_channel = 3
                features = torch.reshape(features, (config["batch_size"], in_channel, window_size)).to(device)
                output = model(features)[:, 0]
                targets = torch.norm(targets, dim=1).to(device)
                for i in range(config["batch_size"]):
                    distance[i] = torch.sum(targets[i * window_size:(i + 1) * window_size])
                total_number += config["batch_size"]
                distance = distance.to(device)
                loss = criterion(output, distance)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))
    print("Finished Training")
    '''
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(np.arange(0, num_epochs), total_loss)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs. Epoch")
    ax1.grid()
    plt.savefig("./our_checkpoints/graphs_" + save_string + ".png")
    plt.show()
    '''

    state = torch.load("./our_checkpoints/" + save_string + "_ckpt.pth", map_location=device)  # there is a _best version
    model.load_state_dict(state['net'])


def second_train(config, train_ds, save_string, curr_weights, abs_value=False, window_size=50, num_epochs=5,
                 data_dir=None, checkpoint_dir=None):
    model = SimpleNet(window_size, curr_weights.get('conv1.bias').shape[0], curr_weights.get('conv2.bias').shape[0],
                      abs_flag=abs_value)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))

    model.conv1.weight.requires_grad = False
    model.conv1.bias.requires_grad = False

    model.conv2.weight.requires_grad = False
    model.conv2.bias.requires_grad = False

    model.norm1.weight.requires_grad = False
    model.norm1.bias.requires_grad = False

    model.norm2.weight.requires_grad = False
    model.norm2.bias.requires_grad = False

    # device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    # loss criterion
    criterion = nn.MSELoss()
    # train and validation
    train_set_size = int(len(train_ds) * 0.8)
    train_subset, val_subset = random_split(train_ds, [train_set_size, len(train_ds) - train_set_size])
    train_loader = DataLoader(train_subset, batch_size=int(config["batch_size"] * window_size), shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=int(config["batch_size"] * window_size), shuffle=False)
    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], momentum=0.9)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    epoch_time = time.time()

    best_loss = 50
    total_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        # training loss
        total_number = 0
        epoch_loss = []
        epoch_time = time.time()
        for features, targets in train_loader:
            distance = torch.zeros(config["batch_size"])
            if np.shape(features)[0] != window_size * config["batch_size"]:
                continue
            if abs_value:
                in_channel = 1
            else:
                in_channel = 3
            features = torch.reshape(features, (config["batch_size"], in_channel, window_size)).to(device)
            output = model(features)[:, 0]
            targets = torch.norm(targets, dim=1).to(device)
            for i in range(config["batch_size"]):
                distance[i] = torch.sum(targets[i * window_size:(i + 1) * window_size])
                if total_number == 5 * config["batch_size"]:
                    res = ' Real Value: {:.4f} | Predicted Value: {:.4f} |'.format(distance[i], output[i])
                    print(res)
            total_number += config["batch_size"]
            distance = distance.to(device)
            loss = criterion(output, distance)
            optimizer.zero_grad()  # clean the gradients from previous iteration
            loss.backward()  # autograd backward to calculate gradients
            optimizer.step()  # apply update to the weights
            epoch_loss.append(loss.item())
        epoch_time = time.time() - epoch_time
        total_loss[epoch] = np.mean(epoch_loss)
        log = 'Epoch: {} | Loss: {:.4f} |'.format(epoch, total_loss[epoch])
        log += ' Epoch Time: {:.2f} secs |'.format(epoch_time)
        log += ' Learning Rate: {:.5f} |'.format(optimizer.param_groups[0]['lr'])
        print(log)
        # scheduler.step()

        if total_loss[epoch] < best_loss:
            print('==> Saving model ...')
            best_loss = total_loss[epoch]
            state = {
                'net': model.state_dict(),
                'epoch': epoch,
            }
            '''
            if not os.path.isdir('our_checkpoints'):
                os.mkdir('our_checkpoints')
            torch.save(state, "./our_checkpoints/" + save_string + "_ckpt.pth")
            '''

        # validation loss
        val_loss = 0.0
        val_steps = 1
        total = 0
        for features, targets in val_loader:
            with torch.no_grad():
                distance = torch.zeros(config["batch_size"])
                if np.shape(features)[0] != window_size * config["batch_size"]:
                    continue
                if abs_value:
                    in_channel = 1
                else:
                    in_channel = 3
                features = torch.reshape(features, (config["batch_size"], in_channel, window_size)).to(device)
                output = model(features)[:, 0]
                targets = torch.norm(targets, dim=1).to(device)
                for i in range(config["batch_size"]):
                    distance[i] = torch.sum(targets[i * window_size:(i + 1) * window_size])
                total_number += config["batch_size"]
                distance = distance.to(device)
                loss = criterion(output, distance)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))
    print("Finished Training")
    '''
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(np.arange(0, num_epochs), total_loss)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs. Epoch")
    ax1.grid()
    plt.savefig("./our_checkpoints/graphs_" + save_string + ".png")
    plt.show()
    '''

    # state = torch.load("./our_second_checkpoints/" + save_string + "_ckpt.pth",
    #                    map_location=device)  # there is a _best version
    # model.load_state_dict(state['net'])


def test_accuracy(features, targets, window_size, model, device="cpu",
                  abs_value=False, criterion=nn.MSELoss(), mode=False):
    if abs_value:
        in_channel = 1
    else:
        in_channel = 3
    num = int(np.floor(np.shape(features)[0] / window_size))
    test_error = 0
    total_dis = 0
    total_out = 0
    output = []
    label = []
    with torch.no_grad():
        for i in range(num):
            test_outputs = model(torch.reshape(torch.from_numpy(features[i * window_size:window_size * (i + 1)]),
                                               (1, in_channel, window_size)).float().to(device))
            output.append(test_outputs)
            check_pos_new = torch.max(torch.sum(torch.norm(torch.from_numpy(targets), dim=1).float()
                                      [i * window_size:window_size * (i + 1)]), torch.tensor(0.0001))
            label.append(check_pos_new)
            log = 'Step: {} | Output: {:.4f} |'.format(i, test_outputs.item())
            log += ' New position: {:.4f} |'.format(check_pos_new)
            log += ' Relative error: {:.4f}% |'.format(torch.abs(test_outputs.item() - check_pos_new) * 100 / check_pos_new)
            print(log)
            test_error += criterion(torch.squeeze(test_outputs), check_pos_new.to(device))
            total_dis = total_dis + check_pos_new
            total_out = total_out + test_outputs.item()
    total_error = torch.abs(total_out - total_dis) * 100 / total_dis
    res_string = 'test MSE error: {:.4f} | Relative error: {:.4f}% |'.format(test_error.item()/num, total_error)
    res_string = "\n" + res_string
    print(res_string)
    f = open("finite_results.txt", "a")
    f.write(res_string)
    f.close()

    return total_error, output, label


def data_acquire(train_test_option, combination_method, window_size):
    train_acce, train_pos = [], []
    test_acce, test_pos = [], []
    rec_test_index = random.sample(list(range(1, 11)), 2)
    train_test_string = [k for k, v in train_test_option.items() if v is True]
    comb_string = [k for k, v in combination_method.items() if v is True]
    save_string = "_".join(train_test_string + comb_string)

    if train_test_option["rec_train"]:
        # the rest of the recordings besides the one used for test
        train_index = [elem for elem in range(1, 11) if elem not in rec_test_index]
        # recordings amount used for train
        if combination_method["voting"]:
            if combination_method["abs_value"]:
                train_acce = [pd.read_csv(
                    os.path.join(os.getcwd(), "Processed", "voting_abs", f"voting_abs_method_file{elem}.csv"),
                    header=None) for elem in train_index]
            else:
                train_acce = [
                    pd.read_csv(os.path.join(os.getcwd(), "Processed", "voting", f"voting_method_file{elem}.csv"),
                                header=None) for elem in train_index]
        if combination_method["average"]:
            if combination_method["abs_value"]:
                train_acce = [
                    pd.read_csv(os.path.join(os.getcwd(), "Processed", "avg_abs", f"avg_abs_method_file{elem}.csv"),
                                header=None) for elem in train_index]
            else:
                train_acce = [pd.read_csv(os.path.join(os.getcwd(), "Processed", "avg", f"avg_method_file{elem}.csv"),
                                          header=None) for elem in train_index]
        else:  # using a single imu
            imu_num = int(combination_method["single_imu"])
            if combination_method["abs_value"]:
                train_acce = [pd.read_csv(
                    os.path.join(os.getcwd(), "Processed", "abs_pre_pick_mimu", f"abs_acc_method_file{elem}.csv"),
                    header=None, usecols=[imu_num]) for elem in train_index]
            else:
                train_acce = [
                    pd.read_csv(os.path.join(os.getcwd(), "Recordings", "second", f"acce_data{elem}.csv"),
                                header=None, usecols=[imu_num*3, imu_num*3+1, imu_num*3+2]) for elem in train_index]

        train_pos = [pd.read_csv(os.path.join(os.getcwd(), "Recordings", "second", "position_modified",
                                              f"position_step{elem}_modified.csv"), header=None) for elem in
                     train_index]
        train_pos = [pd.DataFrame(data=elem) for elem in train_pos]
        train_pos = [elem.div([1000, 1000, 1000]) for elem in train_pos]
        diff_size = [train_acce[cnt].shape[0] - train_pos[cnt].shape[0] for cnt in range(len(train_index))]
        train_acce = [train_acce[cnt][diff_size[cnt]:] for cnt in range(len(train_index))]

    if train_test_option["ridi_train"]:
        dir_path = os.path.join(os.getcwd(), "modified_ridi")
        files = os.listdir(dir_path)
        data_path = [os.path.join(dir_path, elem) for elem in files]
        if combination_method["abs_value"]:
            cols = ["abs_acc"]
            acce_dir_path = os.path.join(os.getcwd(), "Processed", "abs_ridi")
            acce_files = os.listdir(acce_dir_path)
            acce_data_path = [os.path.join(acce_dir_path, elem) for elem in acce_files]
        else:
            cols = ['acce_x', 'acce_y', 'acce_z']
            acce_data_path = data_path
        x = [pd.read_csv(elem, usecols=cols) for elem in acce_data_path]
        y = [pd.read_csv(elem, usecols=['pos_x', 'pos_y', 'pos_z']) for elem in data_path]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

        train_acce = train_acce + x_train
        train_pos = train_pos + y_train

    if train_test_option["ridi_test"]:
        test_acce += train_acce + x_test
        test_pos += train_pos + y_test

    if train_test_option["rec_test"]:
        if combination_method["voting"]:
            if combination_method["abs_value"]:
                test = [pd.read_csv(
                    os.path.join(os.getcwd(), "Processed", "voting_abs", f"voting_abs_method_file{elem}.csv"),
                    header=None) for elem in rec_test_index]
            else:
                test = [
                    pd.read_csv(os.path.join(os.getcwd(), "Processed", "voting", f"voting_method_file{elem}.csv"),
                                header=None) for elem in rec_test_index]
        if combination_method["average"]:
            if combination_method["abs_value"]:
                test = [
                    pd.read_csv(os.path.join(os.getcwd(), "Processed", "avg_abs", f"avg_abs_method_file{elem}.csv"),
                                header=None) for elem in rec_test_index]
            else:
                test = [pd.read_csv(os.path.join(os.getcwd(), "Processed", "avg", f"avg_method_file{elem}.csv"),
                                    header=None) for elem in rec_test_index]
        else:  # using a single imu
            imu_num = int(combination_method["single_imu"])
            if combination_method["abs_value"]:
                test = [pd.read_csv(
                    os.path.join(os.getcwd(), "Processed", "abs_pre_pick_mimu", f"abs_acc_method_file{elem}.csv"),
                    header=None, usecols=[imu_num]) for elem in rec_test_index]
            else:
                test = [
                    pd.read_csv(os.path.join(os.getcwd(), "Recordings", "second", f"acce_data{elem}.csv"),
                                header=None, usecols=[imu_num*3, imu_num*3+1, imu_num*3+2]) for elem in rec_test_index]

        test_label = [pd.read_csv(os.path.join(os.getcwd(), "Recordings", "second", "position_modified",
                                               f"position_step{elem}_modified.csv"), header=None) for elem in
                      rec_test_index]
        test_label = [pd.DataFrame(data=elem) for elem in test_label]
        test_label = [elem.div([1000, 1000, 1000]) for elem in test_label]

        diff_size = [test[cnt].shape[0] - test_label[cnt].shape[0] for cnt in range(len(rec_test_index))]
        test = [test[cnt][diff_size[cnt]:] for cnt in range(len(rec_test_index))]

        test_acce = test_acce + test
        test_pos = test_pos + test_label

    test_acce_cut = [elem if (elem.shape[0] % window_size) == 0 else
                     elem.iloc[:-(elem.shape[0] % window_size)] for elem in test_acce]
    test_pos_cut = [elem if (elem.shape[0] % window_size) == 0 else
                    elem.iloc[:-(elem.shape[0] % window_size)] for elem in test_pos]
    acce_test = np.concatenate(test_acce_cut)
    pos_test = np.concatenate(test_pos_cut)

    train_acce_cut = [elem if (elem.shape[0] % window_size) == 0 else
                      elem.iloc[:-(elem.shape[0] % window_size)] for elem in train_acce]
    train_pos_cut = [elem if (elem.shape[0] % window_size) == 0 else
                     elem.iloc[:-(elem.shape[0] % window_size)] for elem in train_pos]
    acce_train = np.concatenate(train_acce_cut)
    pos_train = np.concatenate(train_pos_cut)

    acce_train = acce_train[1:]
    acce_test = acce_test[1:]
    pos_train = pos_train[1:] - pos_train[:-1]
    pos_test = pos_test[1:] - pos_test[:-1]
    train_ds = TensorDataset(torch.from_numpy(acce_train).float(), torch.from_numpy(pos_train).float())
    save_string += '_' + str(window_size) + '_window_size-for_results'  # change according to wanted file
    return train_ds, acce_test, pos_test, save_string


def main(num_samples=10, max_num_epochs=50):

    # train and test options
    train_test_options = {
        "rec_train": False,
        "ridi_train": True,
        "rec_test": False,
        "ridi_test": True
    }

    # combination method for mimu
    combination_methods = {
        "voting": False,
        "average": False,
        "abs_value": True
    }
    for elem in np.arange(50, 110, 10):
        window_size = elem
        num_epochs = 50
        train_ds, acce_test, pos_test, save_string = data_acquire(train_test_options, combination_methods, window_size)

        init(object_store_memory=8*10**7, num_cpus=1, num_gpus=1, _memory=5*10**9)
        config = {
            "out1": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
            "out2": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
            "lr": tune.loguniform(1e-6, 1e-4),
            "batch_size": tune.choice([2, 4, 8, 16, 32, 64])
        }

        reporter = CLIReporter(
            metric_columns=["loss", "training_iteration"])
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)
        result = tune.run(
            partial(train_data, train_ds=train_ds, save_string=save_string, abs_value=combination_methods["abs_value"],
                    window_size=window_size, num_epochs=num_epochs, data_dir=os.path.join(os.getcwd(), "data_dir"),
                    checkpoint_dir=os.path.join(os.getcwd(), "checkpoints_dir")),
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir="./checkpoints_dir",
            name=save_string,
            reuse_actors=True,
            log_to_file=True,
            resources_per_trial={'gpu': 1, 'cpu': 1})

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))

        best_trained_model = SimpleNet(window_size, best_trial.config["out1"], best_trial.config["out2"],
                                       abs_flag=combination_methods["abs_value"])
        # best_trained_model = SimpleNet(best_trial.config["out1"], best_trial.config["out2"], abs_flag=abs_value)
        device = torch.device("cpu")
        # if torch.cuda.is_available():
        #     device = "cuda:0"
        best_trained_model.to(device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)
        test_error, test_outputs, check_pos_new = test_accuracy(acce_test, pos_test, window_size,
                                                                best_trained_model, device="cpu",
                                                                abs_value=combination_methods["abs_value"])
        print("Best trial test set relative error: {}".format(test_error))
        plt.plot(np.arange(0, len(test_outputs)), test_outputs)
        plt.plot(np.arange(0, len(test_outputs)), check_pos_new)
        plt.xlabel("Step Number")
        plt.ylabel("Position")
        plt.title(f"Test Output and Real Position for {window_size} steps")
        plt.legend(["Test Output", "Real Position"])
        plt.grid()
        plt.savefig("./our_checkpoints/graphs_" + save_string + ".pdf")
        plt.clf()
        shutdown()
        torch.save(best_trained_model.state_dict(), MODEL_WEIGHTS_PATH)

        # ******************************* #
        # ******************************* #
        # ******************************* #
        combination_methods = {
            "voting": False,
            "average": False,
            "abs_value": True,
            "single_imu": 0
        }

        second_train_options = {
            "rec_train": True,
            "ridi_train": False,
            "rec_test": True,
            "ridi_test": False
        }
        window_size = window_size
        num_epochs = 10
        train_ds, acce_test, pos_test, save_string = data_acquire(second_train_options, combination_methods, window_size)
        saved_weights = torch.load(MODEL_WEIGHTS_PATH)
        init(object_store_memory=8 * 10 ** 7, num_cpus=1, num_gpus=1, _memory=5 * 10 ** 9)
        config = {
            "lr": tune.loguniform(1e-6, 1e-2),
            "batch_size": tune.choice([4, 8, 16, 32])
        }

        reporter = CLIReporter(
            metric_columns=["loss", "training_iteration"])
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=num_epochs,
            grace_period=1,
            reduction_factor=2)
        result = tune.run(
            partial(second_train, train_ds=train_ds, save_string=save_string, curr_weights=saved_weights,
                    abs_value=combination_methods["abs_value"], window_size=window_size, num_epochs=num_epochs,
                    data_dir=os.path.join(os.getcwd(), "data_dir"),
                    checkpoint_dir=os.path.join(os.getcwd(), "second_checkpoints_dir")),
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir="./second_checkpoints_dir",
            name=save_string,
            reuse_actors=True,
            resources_per_trial={'gpu': 1, 'cpu': 1})

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))

        best_trained_model = SimpleNet(win_size=window_size, out1=saved_weights.get('conv1.bias').shape[0],
                                       out2=saved_weights.get('conv2.bias').shape[0],
                                       abs_flag=combination_methods["abs_value"])
        # best_trained_model = SimpleNet(best_trial.config["out1"], best_trial.config["out2"], abs_flag=abs_value)
        device = torch.device("cpu")
        # if torch.cuda.is_available():
        #     device = "cuda:0"
        best_trained_model.to(device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)
        test_error, test_outputs, check_pos_new = test_accuracy(acce_test, pos_test, window_size,
                                                                best_trained_model, device="cpu",
                                                                abs_value=combination_methods["abs_value"], mode=True)
        print("Best trial test set relative error: {}".format(test_error))
        plt.plot(np.arange(0, len(test_outputs)), test_outputs)
        plt.plot(np.arange(0, len(test_outputs)), check_pos_new)
        plt.xlabel("Step Number")
        plt.ylabel("Position")
        plt.title(f"Test Output and Real Position")
        plt.legend(["Test Output", "Real Position"])
        plt.grid()
        plt.savefig("./our_second_checkpoints/graphs_" + save_string + ".pdf")
        plt.clf()
        shutdown()


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=15)
