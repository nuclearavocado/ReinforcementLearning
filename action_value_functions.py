"""
  Implements various approaches to estimating the action value function for
  calculating the policy gradient, as seen in Section 2. of the GAE paper:
  https://arxiv.org/pdf/1506.02438.pdf
      - Trajectory-based policy gradient (total reward)
      - Reward-to-go policy gradient
      - Baseline
      - State-action value function (Q-function)
      - Advantage function
  Discounting with gamma, or no discounting with gamma = 1
"""
import numpy as np
import scipy.signal

def discount(rewards, gamma):
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + gamma*R
        returns.append(R)
    return np.asarray(returns[::-1])

def total_reward(rewards, gamma=1.0):
    ep_ret, ep_len = sum(rewards), len(rewards)
    undiscounted_returns = [ep_ret] * ep_len # the weight for each logprob(a|s) is R(tau)
    returns = []
    for i, r in enumerate(undiscounted_returns):
        R = gamma**i * r
        returns.append(R)
    return np.asarray(returns)

def rewards_to_go(rewards, gamma=1.0, value_estimate=False):
    # Batch
    if isinstance(rewards, list):
        for i, reward in enumerate(rewards):
            delta = discount(reward, gamma)
            rewards[i] = delta[:-1] if value_estimate else delta # Ignore final reward for bootstrapping value estimate
        return rewards
    # No batch
    else:
        returns = discount(rewards, gamma)
        return np.asarray(returns)

def baseline():
    pass

def q_function():
    pass

def q_value_function(returns, values, gamma=1.0):
    # Batch
    if isinstance(returns, list):
        q_values = []
        for ret, value in zip(returns, values):
            q_value = ret[:-1] + gamma*value[1:]
            q_values.append(q_value)
        return q_values
    # No batch
    else:
        q_values = returns[:-1] + gamma*values[1:]
        return np.asarray(q_values)

def advantage_function(q_values, values, gamma=1.0):
    # Batch
    if isinstance(q_values, list):
        advantages = []
        for q_value, value in zip(q_values, values):
            advantage = q_value - value[:-1]
            advantage = discount(advantage, gamma)
            advantages.append(advantage)
        return advantages
    # No batch
    else:
        advantages = q_values - values[:-1]
        advantages = discount(advantages, gamma)
        return np.asarray(advantages)

def GAE_Lambda(rewards, values, gamma=0.99, lam=0.95):
    """
    Uses rewards and value estimates from the whole trajectory to compute
    advantage estimates with GAE-Lambda, as well as compute the rewards-to-go
    for each state, to use as the targets for the value function.

    "last_val" = 0 if the trajectory ended because the agent reached a terminal
    state (died); otherwise V(s_T): the value function estimated for the last
    state. This allows bootstraping of the reward-to-go to account for timesteps
    beyond the arbitrary episode horizon (or epoch cutoff).
    """
    # Batch
    if isinstance(rewards, list):
        advantages = []
        for reward, value in zip(rewards, values):
            # print(reward.shape, value.shape)
            deltas = reward[:-1] + gamma*value[1:] - value[:-1]
            advantage = discount(deltas, gamma*lam)
            advantages.append(advantage)
        return advantages
    # No batch
    else:
        deltas = rewards[:-1] + gamma*values[1:] - values[:-1]
        advantages = discount(deltas, gamma*lam)
        return advantages
