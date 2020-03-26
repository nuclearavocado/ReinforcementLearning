'''
    Algorithm:      REINFORCE (REward Increment = Nonnegative Factor times
                    Offset Reinforcement times Characteristic Eligibility)
    Paper:          https://doi.org/10.1007/BF00992696
    Paper author:   Ronald J. Williams
    Institution:    Northeastern University

    TODO:   Implement baselines

    # Baselines
    # self.baseline, self.sum, self.len = 0, 0, 0

    # def _recompute_baseline(self, returns):
    #     self.sum += sum(returns)
    #     self.len += len(returns)
    #     return self.sum/self.len
'''

# Libraries
import torch
import torch.optim as optim
# Local
import action_value_functions
from agents.base_agent import BaseAgent
from utils import normalize

class REINFORCE(BaseAgent):
    def __init__(self, args, env, env_params, logger, buffer, policy, actor):
        super().__init__(args, env, env_params, logger, buffer, policy)
        self.actor = actor
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=5e-3)

    # ----------------------------------------------------------
    # Functions to be overloaded from the parent class

    def _get_nn_outputs(self, obs):
        # Select action a
        actions = self.policy.sample_actions(self.actor, obs)
        output = {'actions': actions}
        return output

    def _finish_trajectory(self):
        pass

    def _logs(self):
        super()._logs()

    def _process_trajectories(self, mb):
        # Compute the returns
        returns = self._compute_returns(mb['rewards'], method='rtg')
        mb = {'observations': mb['observations'],
              'actions': mb['actions'],
              'returns': returns}
        return mb

    def _update_networks(self, mb, retain_graph=False):
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss = -self._compute_actor_loss(mb['observations'], mb['actions'],
                                               action_values=mb['returns']) # *-1 => gradient *ascent*
        actor_loss.backward(retain_graph=retain_graph) # compute gradients
        self.actor_optimizer.step()

    # ----------------------------------------------------------
    # Algorithm specific functions

    def _compute_returns(self, rewards, method='rtg', baseline=None, value_estimate=False):
        if method == 'trajectory': # trajectory-based policy gradient
            returns = action_value_functions.total_reward(rewards.copy(), gamma=self.args.gamma)
        elif method == 'rtg': # reward-to-go policy gradient, discounted by gamma
            returns = action_value_functions.rewards_to_go(rewards.copy(), gamma=self.args.gamma, value_estimate=value_estimate)
        else:
            raise NotImplementedError
        if baseline is not None:
            returns = returns - baseline
        returns = normalize(returns)
        return returns

    def _compute_actor_loss(self, obs, actions, action_values=None):
        '''
            Calculate the policy gradient.
            
            For REINFORCE, the `action_values` are equal to the returns:
                q_π(s,a) = r(s,a)

            Args:
            `obs`           is a tensor of the observations from the mini-batch
                            of trajectories.
            `actions`       is a tensor of the actions from the mini-batch of
                            trajectories.
            `action_values` is a tensor of the values of the actions taken in
                            the mini-batch of trajectories.
                            There are several possible choices of
                            `action_values`. See the paper on Generalized
                            Advantage Estimation (GAE) for more details on the
                            different choices of `action_values`:
                            https://arxiv.org/pdf/1506.02438.pdf
        '''
        # Get log-probability of each action
        log_probs, entropy = self.policy.evaluate_actions(self.actor, obs, actions)
        # Log the entropy
        self.logger.add_scalar('entropy', entropy)

        # Compute loss
        '''
        Policy gradient:
            ∇_θ J(π_θ) = E_{τ~π_θ}[ Σ_{0}^{T} ∇_θ log π_θ(a,s) * q_π(s,a) ]

        Because the policy gradient is an expectation, we can estimate the
        policy gradient with a sample mean:
            ∇_θ J(π_θ) ~ g^hat

        Where for a single sample estimate:
            g^hat = Σ_{t = 0}^{T} ∇_θ log π_θ(a,s) * q_π(s,a)

        And for a set of N trajectories (a batch), D:
            D = {τ_i}, and i = 1, ..., N
        Thus:
            g^hat = 1/|D| Σ_{τ ∈ D} Σ_{t = 0}^{T} ∇_θ log π_θ(a,s) * q_π(s,a)
        '''
        policy_gradient = (log_probs * action_values).mean()
        return policy_gradient
