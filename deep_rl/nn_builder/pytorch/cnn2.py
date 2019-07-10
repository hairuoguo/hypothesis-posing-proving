import torch
import torch.nn as nn
import numpy as np
from deep_rl.nn_builder.pytorch.Base_Network import Base_Network
import torch.nn.functional as F

class CNN(nn.Module):
    ''' A 1D conv network for ReverseEnv '''
    def __init__(self, input_dim, output_dim, config_dict):
        super(CNN, self).__init__()
        self.input_dim = input_dim

        self.y_range = config_dict['y_range']
        self.batch_norm = config_dict['batch_norm']
        self.num_conv_layers = config_dict['num_conv_layers']
        # conv weights shape = (out_channels, in_channels, kernel_size)

        # for kernel_size = 3, should have one padding to keep things the same.
        # do I want the dimension to increase? Can't just increase filters. I
        # don't think Conv net is good to mapping to larger space

        # L_out = L_in + 2*padding - kernel_size + 1
        self.conv1 = torch.nn.Conv1d(1, 10, 3, stride=1,padding=1,bias=True)
        self.conv_layers = [self.conv1] + [torch.nn.Conv1d(10, 10, 3, stride=1,
            padding=1,bias=True) for _ in range(self.num_conv_layers)]
        self.fc1 = torch.nn.Linear(10*input_dim, 1000)
        self.fc2 = torch.nn.Linear(1000, output_dim)

    def forward(self, x):
        # reshape to (batch_size, num_channels, length)
        x = x.view(-1, 1, self.input_dim)
        for layer in self.conv_layers:
            x = F.relu(layer(x))

        x = x.view(-1, 10*self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.y_range:
            x = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x)
        return x
