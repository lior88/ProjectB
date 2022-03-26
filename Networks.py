import torch.nn as nn
import torch
import numpy as np


def calc_conv_dim(in_size, stride, padding, kernel_size, dilate):
    out_size = (in_size + 2*padding - dilate*(kernel_size - 1) - 1)/stride
    out_size = int(np.floor(out_size + 1))
    return out_size


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# not ready


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(in_channels=planes, out_channels=planes * self.expansion, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, abs_flag=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if abs_flag:
            in_channels = 1
        else:
            in_channels = 3

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 224x224
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 112x112

        x = self.layer1(x)  # 56x56
        x = self.layer2(x)  # 28x28
        x = self.layer3(x)  # 14x14
        x = self.layer4(x)  # 7x7
        avgpool_not_200 = nn.AvgPool1d(x.shape[2], stride=1)
        x = avgpool_not_200(x)  # 1x1
        #x = self.avgpool(x)  # 1x1
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class SimpleNet(nn.Module):

    def __init__(self, win_size, out1=32, out2=64, abs_flag=False):

        super(SimpleNet, self).__init__()
        if abs_flag:
            in_planes = 1
        else:
            in_planes = 3

        self.conv1 = nn.Conv1d(in_channels=in_planes, out_channels=out1, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm1d(out1)
        self.conv2 = nn.Conv1d(in_channels=out1, out_channels=out2, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(out2)
        self.relu3 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.norm5 = nn.BatchNorm1d(out2)
        self.drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(int(np.floor(win_size/2) * out2), 16)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu3(x)
        x = self.pool(x)
        # x = self.norm5(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        # x = self.sig(x)

        return x
