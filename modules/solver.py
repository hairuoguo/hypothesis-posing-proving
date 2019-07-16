import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F



class Solver(nn.Module):
    """
    Module which puts together the attention and all conv modules to create a
    solver for the Reverse Environment.

    General procedure:

    """

    def __init__(self, input_dim, output_dim):
        super(Solver, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_net = None # TODO
        self.pointer_net = None # TODO
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

        layer_outputs = self.conv_net(x)
        softmax_out = self.pointer_net(layer_outputs, x)
        centered_input = x - 0.5
        assert softmax_out.shape == centered_input.shape, 'softmax shape: {} does not match input shape: {}'.format(
                softmax_out.shape, centered_input.shape)
        masked_input = softmax_out * centered_input
        out = self.controller(masked_input)
        return out
