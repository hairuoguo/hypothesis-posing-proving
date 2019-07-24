import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, y_range=()):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 100
        self.y_range = y_range
        self.conv1 = nn.Conv1d(1, 10, 3, padding=1)
        self.conv2 = nn.Conv1d(10, 10, 3, padding=1)
        self.conv3 = nn.Conv1d(10, 10, 3, padding=1)
        self.conv4 = nn.Conv1d(10, 10, 3, padding=1)
        self.fc1 = nn.Linear(self.input_dim*10, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)


    def forward(self, x):
        # reshape to (batch_size, num_channels, length)
        x = x.view(-1, 1, self.input_dim)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4_1 = F.relu(self.conv4(x3))
        x4_2 = x4_1.view(-1, self.input_dim*10)
        x5 = F.relu(self.fc1(x4_2))
        x6 = self.fc2(x5) # no activation needed before sigmoid
    
        '''
        if self.y_range:
            x6 = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x6)
        '''
        return x6, torch.cat([x1, x2, x3, x4_1], dim=0).view((-1, 10*4, self.input_dim))
