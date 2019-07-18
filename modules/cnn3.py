import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    ''' A 1D conv network for ReverseEnv '''
    def __init__(self, input_dim, output_dim, y_range=()):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 100
        self.y_range = y_range
        self.conv1 = torch.nn.Conv1d(1, 10, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(10, 10, 3, padding=1)
        self.conv3 = torch.nn.Conv1d(10, 10, 3, padding=1)
        self.conv4 = torch.nn.Conv1d(10, 10, 3, padding=1)
        self.conv5 = torch.nn.Conv1d(10, 10, 3, padding=1)
        self.conv6 = torch.nn.Conv1d(10, 10, 3, padding=1)
        self.conv7 = torch.nn.Conv1d(10, 10, 3, padding=1)
        self.fc1 = torch.nn.Linear(10*self.hidden_dim, self.output_dim)

    def forward(self, x):
        # reshape to (batch_size, num_channels, length)
        x = x.view(-1, 1, self.input_dim)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
    x = F.relu(self.conv7(x))

        x = F.adaptive_avg_pool1d(x, self.hidden_dim)
        x = x.view(-1, 10*self.hidden_dim)
        x = self.fc1(x)

        if self.y_range:
            x = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x)

        return x
