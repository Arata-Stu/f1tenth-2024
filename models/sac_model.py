import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

NN_LAYER_1 = 200
NN_LAYER_2 = 100
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6

class SACPolicyNet(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(SACPolicyNet, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, NN_LAYER_1)
        self.linear2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.mean_linear = nn.Linear(NN_LAYER_2, num_actions)
        self.log_std_linear = nn.Linear(NN_LAYER_2, num_actions)
    
    def forward(self, state):
        
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(0, 1) # assumes actions have been normalized to (0,1)
        
        z = mean + std * normal.sample().requires_grad_()
        action = torch.tanh(z)
        log_prob = torch.distributions.Normal(mean, std).log_prob(z) - torch.log(1 - action * action + EPSILON) 
        log_prob = log_prob.sum(-1, keepdim=True)
            
        return action, log_prob
    

    
class SACCriticNet(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(SACCriticNet, self).__init__()
        # クリティック1
        self.fc1_q1 = nn.Linear(state_dim + act_dim, NN_LAYER_1)
        self.fc2_q1 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_out_q1 = nn.Linear(NN_LAYER_2, 1)
        
        # クリティック2
        self.fc1_q2 = nn.Linear(state_dim + act_dim, NN_LAYER_1)
        self.fc2_q2 = nn.Linear(NN_LAYER_1, NN_LAYER_2)
        self.fc_out_q2 = nn.Linear(NN_LAYER_2, 1)

    def forward(self, state, action):
        # 状態と行動の結合
        x = torch.cat([state, action], dim=1)
        
        # クリティック1のフォワードパス
        q1 = F.relu(self.fc1_q1(x))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc_out_q1(q1)
        
        # クリティック2のフォワードパス
        q2 = F.relu(self.fc1_q2(x))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc_out_q2(q2)
        
        return q1, q2
    
