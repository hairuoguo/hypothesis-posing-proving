import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_dnc_simon.dnc.dnc import DNC


class DNCWrapper(nn.Module):
    def __init__(self, input_dim, output_dim, y_range=(), cuda_index=0):
        super(DNCWrapper, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.y_range = y_range
        self.dnc = DNC(
                input_size=input_dim,
                hidden_size=output_dim, # based on train.py from Hairuo
                rnn_type='lstm',
                num_layers=2,
                num_hidden_layers=2,
                dropout=0,
                nr_cells=16,
                cell_size=20,
                read_heads=4,
                gpu_id=cuda_index,
                debug=False,
                batch_first=True,
                independent_linears=True
        )


    def forward(self, x):
        x = self.dnc(x)
        print('shape1: {}'.format(x))

        if self.y_range:
            x = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x)
        





