import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F



class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, query_size, input_size):

        super(Attention, self).__init__()
            self.query_size = query_size
            self.key_layer1 = nn.Linear(input_size, 2048)
            self.key_layer2 = nn.Linear(2048, query_size)


    def forward(self, input,
                query):
        """
        Attention - Forward-pass
        """
        keys0 = F.relu(self.key_layer1(input))
        keys = self.key_layer2(keys0)
        
        weights = F.softmax(torch.div(torch.matmul(query, keys), math.sqrt(self.query_size))) #TODO: check dimensions
        output = torch.mul(keys, input)
        
        return output
