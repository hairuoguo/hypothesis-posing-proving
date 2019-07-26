import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

class ABCNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=2048, y_range=(), use_lstm=False, device=None):
        super(ABCNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.w = 3
        self.y_range = y_range
        self.conv1 = nn.Conv1d(1, 12, 3, padding=1)
        self.conv2 = nn.Conv1d(12, 12, 3, padding=1)
        self.conv3 = nn.Conv1d(12, 12, 3, padding=1)
        self.conv4 = nn.Conv1d(12, 12, 3, padding=1)
        self.use_lstm = use_lstm

        self.avg_pool = nn.AvgPool1d(self.w, 1, padding=1)
        self.fc1 = nn.Linear(self.input_dim*12, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)

        self.lstm = nn.LSTM(self.input_dim*12, self.hidden_dim)
        if device:
            self.curr_state = (torch.randn((1, 1, self.hidden_dim), device=device), torch.randn((1, 1, self.hidden_dim), device=device))

    def forward(self, x):
        obs, target = torch.chunk(x, 2, dim=1)
        
        obs = obs.view(-1, 1, obs.size(1))
        target = target.view(-1, 1, target.size(1))
        x_obs_1 = F.tanh(self.conv1(obs))
        x_target_1 = F.tanh(self.conv1(target))
        x_obs_1, x_target_1 = self.apply_attention(x_obs_1, x_target_1)
        x_obs_1 = self.weighted_avg_pool(x_obs_1)*self.w
        x_target_1 = self.weighted_avg_pool(x_target_1)*self.w 
        
        x_obs_2 = F.tanh(self.conv2(x_obs_1))
        x_target_2 = F.tanh(self.conv2(x_target_1))
        #x_obs_2, x_target_2 = self.apply_attention(x_obs_2, x_target_2)
        x_obs_2 = self.weighted_avg_pool(x_obs_2)
        x_target_2 = self.weighted_avg_pool(x_target_2)
        
        x_obs_3 = F.tanh(self.conv3(x_obs_2))
        x_target_3 = F.tanh(self.conv3(x_target_2))
        #x_obs_3, x_target_3 = self.apply_attention(x_obs_3, x_target_3)
        x_obs_3 = self.weighted_avg_pool(x_obs_3)
        x_target_3 = self.weighted_avg_pool(x_target_3)
        
        x_obs_4 = F.tanh(self.conv4(x_obs_3))
        x_target_4 = F.tanh(self.conv4(x_target_3))
        #x_obs_4, x_target_4 = self.apply_attention(x_obs_4, x_target_4)
        x_obs_4 = self.weighted_avg_pool(x_obs_4)
        x_target_4 = self.weighted_avg_pool(x_target_4)

        out = torch.cat((x_obs_4.view(x_obs_4.size(0), -1), x_target_4.view(x_target_4.size(0), -1)), dim=1) 


        if self.use_lstm:
            out, hidden = self.lstm(out.unsqueeze(0), self.curr_state)
            self.curr_state = hidden
            out = out.view(-1, self.hidden_dim)
        else: 
            out = F.relu(self.fc1(out))
        out = self.fc2(out)

        if self.y_range:
            out = self.y_range[0] + (self.y_range[1] -
                    self.y_range[0])*nn.Sigmoid()(out) 
        return out

    def weighted_avg_pool(self, x):
        x = self.avg_pool(x)
        #x = torch.mul(x, self.w)
        return x 

    def apply_attention(self, obs_features, target_features):
        attn = self.get_attention_matrix(obs_features, target_features)
        obs_attn = torch.sum(attn, 2)#sum over rows
        obs_features = torch.mul(obs_attn.unsqueeze(1), obs_features)
        target_attn = torch.sum(attn, 1)#sum over columns
        target_features = torch.mul(target_attn.unsqueeze(1), obs_features)
        return obs_features, target_features 

    def get_attention_matrix(self, obs_features, target_features):
        obs_features_m = torch.stack([obs_features]*obs_features.size(2))
        target_features_m = torch.stack([obs_features]*target_features.size(2)).transpose(0, 1)
        return 1./(1. + torch.sum((obs_features_m - target_features)**2, 2)).transpose(0, 1)
           
