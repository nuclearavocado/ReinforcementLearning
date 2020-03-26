'''
    Algorithm:      Actor-Critic
    Paper:          -
    Paper authors:  -
    Institution:    -
'''

# Libraries
import numpy as np
import torch
import torch.optim as optim
# Local
import action_value_functions
from agents.reinforce import REINFORCE
from utils import normalize

class ActorCritic(REINFORCE):
    def __init__(self, args, env, env_params, logger, buffer, policy, actor, critic):
        super().__init__(args, env, env_params, logger, buffer, policy, actor)
        self.critic = critic
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=1e-4)

    # ----------------------------------------------------------
    # Functions to be overloaded from the parent class

    def _get_nn_outputs(self, obs):
        # Estimate the value of state s
        values = self._estimate_value(obs)
        # Select action a
        actions = self.policy.sample_actions(self.actor, obs)
        output = {'actions': actions,
                  'values': values}
        return output

    def _finish_trajectory(self):
        '''
            `last_val` should be the transition reward (typically 0) if the
            trajectory ended because the agent reached a terminal state (died),
            and otherwise should be V(s_T), the value function estimated for the
            last state.
            This allows us to bootstrap the reward-to-go calculation to account
            for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        '''
        last_value = self.buffer.rewards[-1] if self.buffer.dones[-1] else self._estimate_value(self.buffer.obs[-1])
        self.buffer.outputs['values'].append(last_value)
        self.buffer.rewards.append(last_value)

    def _logs(self):
        super()._logs()

    def _process_trajectories(self, mb):
        # Compute the returns
        returns = self._compute_returns(mb['rewards'],
                                        method='rtg')
        # Compute the advantages
        advantages = self._compute_advantages(mb['rewards'], mb['values'])
        mb = {'observations': mb['observations'],
              'actions': mb['actions'],
              'returns': returns,
              'advantages': advantages}
        return mb

    def _update_networks(self, mb, retain_graph=False):
        # Update actor
        for _ in range(self.args.n_pi_updates):
            self.actor_optimizer.zero_grad()
            actor_loss = -self._compute_actor_loss(mb['observations'], mb['actions'],
                                                   action_values=mb['advantages']) # *-1 => gradient *ascent*
            actor_loss.backward(retain_graph=retain_graph) # compute gradients
            self.actor_optimizer.step()
        # Update critic
        for _ in range(self.args.n_v_updates):
            self.critic_optimizer.zero_grad()
            critic_loss = self._compute_critic_loss(mb['observations'], mb['returns'])
            critic_loss.backward(retain_graph=retain_graph) # compute gradients
            self.critic_optimizer.step()

    # ----------------------------------------------------------
    # Algorithm specific functions

    def _estimate_value(self, obs):
        with torch.no_grad():
            # Get the value
            value = self.critic(torch.from_numpy(obs).float())
        return value

    def _compute_returns(self, rewards, method='rtg', baseline=None, value_estimate=False):
        return super()._compute_returns(rewards,
                                        method=method,
                                        baseline=baseline,
                                        value_estimate=value_estimate)

    def _compute_advantages(self, rewards, values):
        '''
            Advantage function
        '''
        q_values = action_value_functions.q_value_function(rewards, values,
                                                           gamma=self.args.gamma)
        advantages = action_value_functions.advantage_function(q_values, values,
                                                               gamma=self.args.gamma)
        return advantages

    def _compute_actor_loss(self, obs, actions, action_values=None):
        '''
            Calculate the policy gradient.
            
            For actor-critic, the `action_values` are equal to the advantages:
                q_π(s,a) = A(s,a)
        '''
        return super()._compute_actor_loss(obs, actions,
                                           action_values=action_values)

    def _compute_critic_loss(self, obs, returns):
        # Get the value
        values = self.critic(obs)
        # Fit value function by regression on mean squared error
        MSE = ((values - returns).pow(2)).mean() # gradient descent
        return MSE
