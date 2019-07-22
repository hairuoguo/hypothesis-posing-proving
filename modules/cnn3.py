import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.cnn import CNN as CNN2

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, y_range=(), num_filters=10):
        super(CNN, self).__init__()
        self.cnn = CNN2(input_dim, output_dim, y_range, num_filters=num_filters)


    def forward(self, x):
        return self.cnn(x)[0]
