import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.resnet import BasicBlock
import math

class AllConv(nn.Module):
    def __init__(self, input_dim, reverse_len, y_range=(), num_filters=10,
            num_blocks=1):
        super().__init__()
        self.input_dim = input_dim

        # for determining output dim. assumes stride of reverse_env is 1.
        self.str_len = int(input_dim / 2)
        self.first_reverse_index = math.floor((reverse_len-1)/2)
        self.last_reverse_index = self.str_len - math.floor(reverse_len/2)
        assert input_dim % 2 == 0, 'should have even length input for reverse_env with HER'

        self.y_range = y_range
        self.num_filters = num_filters

        self.conv1 = nn.Conv1d(1, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.basic_blocks = nn.ModuleList([BasicBlock(num_filters, num_filters)
            for i in range(num_blocks)])

        self.hidden_filter_dim = 512
        self.conv2 = nn.Conv1d(num_filters, self.hidden_filter_dim, 1)
        self.bn2 = nn.BatchNorm1d(self.hidden_filter_dim)
        self.conv3 = nn.Conv2d(self.hidden_filter_dim, self.hidden_filter_dim, kernel_size=(1,2))
        self.bn3 = nn.BatchNorm2d(self.hidden_filter_dim)
        self.conv4 = nn.Conv1d(self.hidden_filter_dim, 1, 1)

    def forward(self, input):
        x = input
        # reshape to (batch_size, num_channels, length)
        x = x.view(-1, 1, self.input_dim)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        for block in self.basic_blocks:
            x = block(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # the 2D conv to convert from being 2*str_len long to 1*str_len long.
        # converts  (batch_size, num_filters, str_len) to 
        #           (batch_size, num_filters, str_len/2) by taking a convolution
        x = x.view(-1, self.hidden_filter_dim, 2, self.str_len).transpose(2, 3)
        x = self.conv3(x) 
        x = self.bn3(x)
        x = F.relu(x)

        x = x.view(-1, self.hidden_filter_dim, self.str_len)

        x = self.conv4(x)
        x = x.view(-1, self.str_len)
        x = x[:, self.first_reverse_index:self.last_reverse_index]


        if self.y_range:
            x = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x)

        return x


class FC2(nn.Module):
    def __init__(self, input_dim, output_dim, y_range=(), batch_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 256
        self.batch_norm = batch_norm
        self.y_range = y_range
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)

        if self.y_range:
            x = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x)

        return x
