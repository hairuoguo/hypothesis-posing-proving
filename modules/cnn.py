import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, y_range=(), num_filters=10):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 100
        self.y_range = y_range
        self.num_filters = num_filters
        self.conv1 = nn.Conv1d(1, num_filters, 3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, 3, padding=1)
        self.conv3 = nn.Conv1d(num_filters, num_filters, 3, padding=1)
        self.conv4 = nn.Conv1d(num_filters, num_filters, 3, padding=1)
        self.fc1 = nn.Linear(self.input_dim*num_filters, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)


    def forward(self, x):
        # reshape to (batch_size, num_channels, length)
        x = x.view(-1, 1, self.input_dim)
        x1 = F.relu(self.conv1(x)) 
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4_1 = F.relu(self.conv4(x3))
        x4_2 = x4_1.view(-1, self.input_dim*self.num_filters)
        #print('shape4_2: {}'.format(x4_2.shape))
        x5 = F.relu(self.fc1(x4_2))
        x6 = self.fc2(x5) # no activation needed before sigmoid
    
        '''
        if self.y_range:
            x6 = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x6)
        '''

        # second val has shape (batch_size, num_layers*num_channels, input_dim)
        return x6, torch.cat([x1, x2, x3, x4_1], dim=1)
