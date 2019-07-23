import torch
import torch.nn as nn
import numpy as np
from deep_rl.nn_builder.pytorch.Base_Network import Base_Network
import torch.nn.functional as F

class CNN(nn.Module):
    ''' A 1D conv network for ReverseEnv '''
    def __init__(self, input_dim, output_dim, y_range=(), num_filters=10):
        super().__init__()
        self.input_dim = input_dim
        self.num_filters = num_filters

        self.y_range = y_range
        conv1 = torch.nn.Conv1d(1, 10, 3, stride=1,padding=1,bias=True)
        bn1 = torch.nn.BatchNorm1d(10)
        self.conv_layers = nn.ModuleList([conv1] + [torch.nn.Conv1d(10, 10, 3,
                stride=1, padding=1,bias=True) for _ in range(self.num_conv_layers)]) 
        fc1 = torch.nn.Linear(10*input_dim, self.linear_hidden_units[0]) 
        self.fc_layers = nn.ModuleList([torch.nn.Linear(
                self.linear_hidden_units[i], self.linear_hidden_units[i+1])
                for i in range(len(self.linear_hidden_units)-1)])


    def forward(self, x):
        # reshape to (batch_size, num_channels, length)
        x = x.view(-1, 1, self.input_dim)

        for layer in self.conv_layers: 
            x = F.relu(layer(x)) 

        x = x.view(-1, 10*self.input_dim)

        for layer in self.fc_layers: 
            x = F.relu(layer(x)) 

        if self.y_range: 
            x = self.y_range[0] + (self.y_range[1] - self.y_range[0])*nn.Sigmoid()(x) 

        return x

