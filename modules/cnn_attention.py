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
        self.conv_net = CNN(input_dim, 256)
        self.attention = Attention(input_dim, input_dim)
        self.fc1 = nn.Linear(input_dim*4*10, 2048)
        #self.fc1 = nn.Linear(input_dim*4*10, 2048)
        self.fc2 = nn.Linear(2048, output_dim)



    def forward(self, x): 
        """
            x will probably have input dimension 20 to start. Then shape is
            (batch_size, 20)

            steps:
            - feed into CNN. CNN outputs tuple of layer outputs, each with the
              same dimension.
            - feed output of CNN into attention network. outputs a softmax.
            - multiply the softmax with a centered version of the input.
            - feed into a small fc network.
            - output a prediction

            Parameters I need:
            - 

        """

        centered_input = x - 0.5
        final_output, layer_outputs = self.conv_net(x)
        out = self.attention(layer_outputs, x)
        #out = self.fc1(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        if self.y_range:
            out = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(out)
        return out
