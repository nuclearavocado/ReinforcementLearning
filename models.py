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

def makeconv(in_shape, out_channels, kernel_size, stride=(1, 1), padding='same'):
    '''
        Padding dimension:
            For (f1, f2) filter applied to (n1, n2) image, the resulting image
            is:
            (n1-f1+1, n2-f2+1)
        Strides
            For (f1, f2) filter applied to (n1, n2) image, with padding (p1, p2)
            and strides (s1, s2), the resulting image is:
            ((n1+2p1-f1)/s + 1, (n2+2p2-f2)/s + 1).
    '''
    n1, n2, nc = in_shape
    f1, f2 = kernel_size
    s1, s2 = stride
    if padding == 'same':
        p1, p2 = 0, 0
        if f1 > 1:
            p1 = 1
        elif f2 > 1:
            p2 = 1
    else:
        raise ValueError
    conv = nn.Conv2d(nc, out_channels, (f1, f2), stride=(s1, s2), padding=(p1, p2))
    size = (n1//s1, n2//s2, out_channels)
    return size, conv

def makeblock(in_shape, out_channels):
    # Downsampling layer
    in_shape, conv1 = makeconv(in_shape, out_channels, (1, 1), padding='same')
    # (1,3) and (2,3)
    in_shape, conv2 = makeconv(in_shape, out_channels, (1, 3), stride=(1, 2), padding='same')
    in_shape, conv3 = makeconv(in_shape, out_channels, (3, 1), stride=(2, 1), padding='same')
    return in_shape, conv1, conv2, conv3


class CNN(nn.Module):
    def __init__(self, env_params):
        super().__init__()
        self.max_action = env_params['max_action']
        # Block1
        in_shape = env_params['resized_image_size']
        print('Architecture:')
        print(f'\tInput:\n\t{in_shape}')
        in_shape, self.conv0, self.conv1, self.conv2 = makeblock(in_shape, (2**4))
        print(f'\tBlock 1 output shape:\n\t{in_shape}')
        # Block2
        # in_shape, self.conv3, self.conv4, self.conv5 = makeblock(in_shape, (2**5))
        in_shape, self.conv3, self.conv4, self.conv5 = makeblock(in_shape, (2**0))
        print(f'\tBlock 2 output shape:\n\t{in_shape}')
        # Block3
        # in_shape, self.conv6, self.conv7, self.conv8 = makeblock(in_shape, (2**6))
        in_shape, self.conv6, self.conv7, self.conv8 = makeblock(in_shape, (2**0))
        print(f'\tBlock 3 output shape:\n\t{in_shape}')
        # Downsampling layer
        # out_channels = (2**8)
        out_channels = (2**0)
        in_shape, self.conv9 = makeconv(in_shape, out_channels, (1, 1), padding='same')
        self.fc1 = nn.Linear(in_shape[0]*in_shape[1]*out_channels, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, env_params['actions'])
        if isinstance(env_params['action_space'], Box): # Continuous
            self.log_sigma = nn.Parameter(torch.zeros(1, env_params['actions']))

    def forward(self, x):
        # Layer 1
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = torch.flatten(x, 1)
        # Layer 1
        x = self.fc1(x)
        x = F.elu(x)
        # Intermediate
        x = self.dropout(x)
        # Layer 2
        x = self.fc2(x)
        return x
