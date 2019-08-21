import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_dnc_simon.dnc.dnc import DNC


class DNCWrapper(nn.Module):
    def __init__(self, input_dim, output_dim, y_range=(), cuda_index=0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.y_range = y_range
        self.hidden_size = 128
        self.dnc = DNC(
                input_size=10, # size of each token
                hidden_size=self.hidden_size,
                output_size=output_dim,
                rnn_type='lstm',
                num_layers=1, # number of RNN layers
                num_hidden_layers=2, # num hidden layers per RNN
                nr_cells=10, # number of memory cells
                cell_size=20,
                read_heads=1,
                gpu_id=cuda_index,
                debug=False,
                batch_first=True, # shape of input tensor is (batch, seq_len, token_dim)
        )

        self.controller_hidden = None        
        self.memory = None
        self.read_vectors = None
        self.reset_experience = True # used for next forward pass


    def reset_experience(self):
        print('hi')
        self.reset_experience = True

    def should_stop_computing(self):
        return True


    def forward(self, x):
        # input should be shape (batch_size, sequence_len, input_size) for the
        # dnc. Sequence_len is RNNed upon.
        x = x.view(-1, self.input_dim, 1)

        print('input {}'.format(x.shape))


        output, (self.controller_hidden, self.memory, self.read_vectors) = self.dnc(
                x, (self.controller_hidden, self.memory, self.read_vectors),
                reset_experience=self.reset_experience)

        self.reset_experience = False
        
        print('output: {}'.format(output.shape))
#        print('controller hidden: {}'.format(self.controller_hidden[0][0].shape))
        output = output[:, -1, :]

        if self.y_range:
            x = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x)

        return x
        






