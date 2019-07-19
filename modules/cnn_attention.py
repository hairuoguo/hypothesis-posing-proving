import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys
from modules.cnn import *
from modules.attention import *



class CNNAttention(nn.Module):
    """
    Module which puts together the attention and all conv modules to create a
    solver for the Reverse Environment.
    """

    def __init__(self, input_dim, output_dim, y_range=()):
        super(CNNAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.y_range = y_range

        self.cnn_output_dim = 256
        self.conv_net = CNN(input_dim, self.cnn_output_dim)
        self.attention = Attention(self.cnn_output_dim, input_dim)
        self.cnn_num_layers = 4
        self.fc1 = nn.Linear(input_dim*self.cnn_num_layers, output_dim)



    def forward(self, x): 
        print('shape1: {}'.format(x.shape))
        centered_input = x - 0.5
        final_output, layer_outputs = self.conv_net(x)
        print('shape2: {}'.format(final_output.shape))
        print('shape3: {}'.format(layer_outputs.shape))

        # layer outputs has dimension [4, batch_size, num_channels, input_dim]
        out = self.attention(layer_outputs, final_output)
        print('shape4: {}'.format(out.shape))
        out = self.fc1(out)
        print('shape5: {}'.format(out.shape))

        if self.y_range:
            out = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(out)

        return out
