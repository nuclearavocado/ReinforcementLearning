import numpy as np
from gym.spaces import Box, Discrete, Dict
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def get_env_spaces(observation_space, action_space):
    act_limit = None
    # Observation space
    if isinstance(observation_space, Discrete):
        obs_dim = observation_space.n
    elif isinstance(observation_space, Box):
        obs_dim = observation_space.shape[0]
    elif isinstance(observation_space, Dict):
        if isinstance(observation_space["observation"], Discrete):
            obs_dim = observation_space["observation"].n + \
                      observation_space["desired_goal"].n + \
                      observation_space["achieved_goal"].n
        elif isinstance(observation_space["observation"], Box):
            obs_dim = observation_space["observation"].shape[0] + \
                      observation_space["desired_goal"].shape[0] + \
                      observation_space["achieved_goal"].shape[0]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    # Action space
    if isinstance(action_space, Discrete):
        act_dim = action_space.n
    elif isinstance(action_space, Box):
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
    else:
        raise NotImplementedError
    return obs_dim, act_dim, act_limit

"""
    Policy Gradient
"""

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = self._log_prob_from_distribution(pi, act) if act is not None else None
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh,
                 image=False, algo=None):
        super().__init__()

        # Get dimensions of observation and action space
        obs_dim, act_dim, _ = get_env_spaces(observation_space, action_space)

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, act_dim, hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, act_dim, hidden_sizes, activation)
        else:
            raise NotImplementedError
        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32) # convert from numpy to tensor
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

"""
    Q-Learning
"""

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCriticQ(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, algo=None):
        super().__init__()
        self.algo = algo

        # Get dimensions of observation and action space
        obs_dim, act_dim, act_limit = get_env_spaces(observation_space, action_space)

        # build Q-function
        if algo in ["td3", "sac"]:
            self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
            self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        else:
            self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        # build policy function
        if algo == "sac":
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        else:
            self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)

    def act(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32) # convert from numpy to tensor
        with torch.no_grad():
            if self.algo == "sac":
                a, _ = self.pi(obs, deterministic, False)
                return a.numpy()
            else:
                return self.pi(obs).numpy()
