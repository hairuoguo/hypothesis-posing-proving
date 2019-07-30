import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_dim, output_dim, y_range=(), cuda_index=0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.y_range = y_range
        self.hidden_size = 256
        self.rnn = nn.LSTM(
                input_size=1, # size of each feature in sequence
                hidden_size=self.hidden_size, 
                num_layers=2,
                batch_first=True, # shape of input tensor is (batch, seq_len, token_dim)
        )
        self.fc1 = nn.Linear(self.hidden_size, self.output_dim)


    def forward(self, x):
        # input should be shape (batch_size, sequence_len, input_size) for the
        # dnc. Sequence_len is RNNed upon.
#         print('input: {}'.format(x.shape))

        x = x.view(-1, self.input_dim, 1)
#         print('input2: {}'.format(x.shape))
        output, (h_n, c_n) = self.rnn(x)
#         print('o: {}'.format(output.shape))
        x = output[:,-1]
#         print('x: {}'.format(x.shape)

        x = self.fc1(x)
#         print('last: {}'.format(x.shape))

        if self.y_range:
            x = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x)

        return x
        






