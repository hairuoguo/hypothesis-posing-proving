import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, conv1_filters, conv2_filters):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(conv1_filters, conv2_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.conv2 = nn.Conv1d(conv2_filters, conv2_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv2_filters)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, y_range=(), num_filters=10,
            num_blocks=1):
        super(ResNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.y_range = y_range
        self.num_filters = num_filters
        self.conv1 = nn.Conv1d(1, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.basic_blocks = nn.ModuleList([BasicBlock(num_filters, num_filters) for i
            in range(num_blocks)])
        self.fc1 = nn.Linear(num_filters*input_dim, output_dim)
#        self.bn2 = nn.BatchNorm1d(256)
#        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, inputs):
        x = inputs
        # reshape to (batch_size, num_channels, length)
        x = x.view(-1, 1, self.input_dim)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        for block in self.basic_blocks:
            x = block(x)

        x = x.view(-1, self.input_dim*self.num_filters)

        x = self.fc1(x)
#        x = self.bn2(x)
#        x = F.relu(x)

#        x = self.fc2(x)

        if self.y_range:
            x = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x)

        return x


class CNNRes(nn.Module):
    def __init__(self, input_dim, output_dim, y_range=(), num_filters=10):
        super(CNNRes, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.y_range = y_range
        self.num_filters = num_filters
        self.f = [self.num_filters]*4
        self.conv1 = nn.Conv1d(1, self.f[0], 3, padding=1)
        self.bn1 = nn.BatchNorm1d(self.f[0])
        self.conv2 = nn.Conv1d(self.f[0], self.f[1], 3, padding=1)
        self.bn2 = nn.BatchNorm1d(self.f[1])
        self.conv3 = nn.Conv1d(self.f[1], self.f[2], 3, padding=1)
        self.bn3 = nn.BatchNorm1d(self.f[2])
        self.conv4 = nn.Conv1d(self.f[2], self.f[3], 3, padding=1)
        self.bn4 = nn.BatchNorm1d(self.f[3])
        self.fc1 = nn.Linear(self.input_dim*self.f[3], 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, self.output_dim)

    def forward(self, x):
        # reshape to (batch_size, num_channels, length)
        x = x.view(-1, 1, self.input_dim)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        identity = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x += identity

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = x.view(-1, self.input_dim*self.f[3])

        x = self.fc1(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.fc2(x)


        if self.y_range:
            x = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x)

        return x


