import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box, Discrete

class MLP(nn.Module):
    def __init__(self, env_params):
        super().__init__()
        self.max_action = env_params['max_action']
        self.fc1 = nn.Linear(env_params['observations'], 128)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, env_params['actions'])
        if isinstance(env_params['action_space'], Box): # Continuous
            self.log_sigma = nn.Parameter(torch.zeros(1, env_params['actions']))

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.elu(x)
        # Layer 2
        x = self.fc2(x)
        return x

class Actor(MLP):
    def __init__(self, env_params):
        super().__init__(env_params)

    def forward(self, x):
        return super().forward(x)

class Critic(nn.Module):
    def __init__(self, env_params):
        super().__init__()
        self.fc1 = nn.Linear(env_params['observations'], 128)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.elu(x)
        # Layer 2
        x = self.fc2(x)
        return x
