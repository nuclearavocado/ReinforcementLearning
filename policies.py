import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class CategoricalPolicy:
    def __init__(self):
        pass

    def sample_actions(model, obs):
        with torch.no_grad():
            # Get the logits from the policy
            logits = model(torch.from_numpy(obs).float())
            # NOTE: Torch lets you get a distribution directly from logits with:
            # distribution = Categorical(logits=logits)
            # but I think applying the softmax here makes it clearer what's happening
            probs = F.softmax(logits, dim=-1) # apply softmax to the final dimension, containing the logits
            distribution = Categorical(probs=probs)
            action = distribution.sample()
            return action.item()

    def evaluate_actions(model, obs, actions):
        '''
            Get log-probability of each action and entropy
        '''
        # Pass mini-batch observations through the model again
        logits = model(obs)
        probs = F.softmax(logits, dim=-1) # apply softmax to the final dimension, containing the logits
        distribution = Categorical(probs=probs)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()
        return log_probs, entropy

class GaussianPolicy:
    def __init__(self):
        pass

    def sample_actions(model, obs):
        # # Handle discrete vs. continuous environments
        # if not self.env_params['discrete_env']: # Continuous
        with torch.no_grad():
            # Get the mean from the policy
            mean = model(torch.from_numpy(obs).float())
            log_std = log_std.expand_as(mean)
            std = torch.exp(model.log_sigma)
            distribution = Normal(mean, std)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions).sum(dim=1, keepdim=True) # gaussian_likelihood
            entropy = distribution.entropy().mean()
            return actions.numpy().squeeze(0), log_probs, entropy
