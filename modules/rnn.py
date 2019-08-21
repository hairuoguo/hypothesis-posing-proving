import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_dim, output_dim, y_range=(), cuda_index=0):
        super().__init__()
        self.input_dim = 2
        self.output_dim = output_dim
        self.y_range = y_range
        self.hidden_size = 256
        self.rnn = nn.LSTM(
                input_size=2, # size of each feature in sequence
                hidden_size=self.hidden_size, 
                num_layers=1,
                batch_first=True, # shape of input tensor is (seq_len, batch, token_dim)
        )
        self.fc1 = nn.Linear(self.hidden_size, self.output_dim)


    def forward(self, x):
        # input should be shape (batch_size, sequence_len, input_size) for the
        # dnc. Sequence_len is RNNed upon.

        assert len(x.shape) == 2 and x.shape[1] % 2 == 0
        x = x.view(x.shape[0], 2, x.shape[1] // 2).transpose(1, 2)
        output, (hn, cn) = self.rnn(x)

        assert hn.shape[0] == 1
        # hn is (num_layers * num_directions, batch, hidden_size)
        x = hn.squeeze(0)  # (batch, hidden_size)

        x = self.fc1(x)

        if self.y_range:
            x = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(x)

        return x
        






