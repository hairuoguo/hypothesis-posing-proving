import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from cnn import *
from attention import *



class CNN-attention(nn.Module):
    """
    Module which puts together the attention and all conv modules to create a
    solver for the Reverse Environment.
    """

    def __init__(self, input_dim, output_dim):
        super(CNN-attention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_net = CNN(input_dim, 256)
        self.attention = Attention(256, input_dim)
        self.controller = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Linear(128, output_dim)
                )



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
        out = self.attention(layer_outputs, final_output)
#        out = F.adaptive_max_pool1d(out, self.output_dim)
        return out
