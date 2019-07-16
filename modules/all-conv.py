import torch as nn
import torch.nn.functional as F

#TODO: should also just try one-layer CNN on implemented models to see if learns.
#TODO: would be nice to have way to differentiate between target and obs

class AllConvNet1D(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet1D, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 4, 1, padding=1)
        self.conv2 = nn.Conv1d()
        self.conv3 = nn.Conv1d()


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        pool_out = F.adaptive_avg_pool1d(50) 


        return conv1_out, conv2_out, pool_out
