import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO: should also just try one-layer CNN on implemented models to see if learns.
#TODO: would be nice to have way to differentiate between target and obs

class AllConvNet1D(nn.Module):
    def __init__(self, input_dim, output_dim, y_range=()):
        super(AllConvNet1D, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 100
        self.y_range = y_range
        self.conv1 = nn.Conv1d(1, 10, 3, padding=1)
        self.conv2 = nn.Conv1d(10, 10, 3, padding=1)
        self.conv3 = nn.Conv1d(10, 10, 3, padding=1)
        self.conv4 = nn.Conv1d(10, 1, 1)
        self.fc1 = nn.Linear(self.hidden_dim, output_dim)


    def forward(self, x):
        # reshape to (batch_size, num_channels, length)
        x = x.view(-1, 1, self.input_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.adaptive_avg_pool1d(x, self.hidden_dim)
        x = x.view(-1, self.hidden_dim)
        x = self.fc1(x) # no activation needed before sigmoid

        if self.y_range:
            x = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x)

        return x
