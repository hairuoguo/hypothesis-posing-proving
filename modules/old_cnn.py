import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, y_range=()):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 100
        self.num_filters=10
        self.y_range = y_range
        self.conv1 = nn.Conv1d(1, self.num_filters, 3, padding=1)
        self.conv2 = nn.Conv1d(self.num_filters, self.num_filters, 3, padding=1)
        self.conv3 = nn.Conv1d(self.num_filters, self.num_filters, 3, padding=1)
        self.fc1 = nn.Linear(self.input_dim*self.num_filters, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)


    def forward(self, x):
        # reshape to (batch_size, num_channels, length)
        x = x.view(-1, 1, self.input_dim)
        x = F.relu(self.conv1(x)) 
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.input_dim*self.num_filters)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # no activation needed before sigmoid

        if self.y_range:
            x = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x)

        return x
