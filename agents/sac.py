# Libraries
import itertools
import numpy as np
import torch
import torch.optim as optim
# Local
from agents.td3 import TD3
from agents.ddpg import DDPG

class SAC(TD3, DDPG):
    """
        Algorithm:      SAC (Soft Actor-Critic)
        Paper:          https://arxiv.org/abs/1801.01290
        Paper authors:  Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel,
                        and Sergey Levine
        Institution:    University of California, Berkeley

        SAC implements three changes to TD3:

        Entropy Regularization
            The Bellman backup now includes an entropy regularization term that
            rewards exploration.

        Learning a Stochastic Policy
            SAC learns a stochastic policy, and thus explores in an on-policy
            way. In addition, the next state actions used for the Bellman
            equation targets are obtained from the current policy, rather than
            deterministicly taking the argmin wrt. actions of a Q-function.

        No Target Policy Smoothing
            Because SAC learns a stochastic rather than deterministic policy,
            there is no need to add additional noise to the policy's output
            action to prevent exploitation of Q-network errors.

        Implementation notes:
            - In the original paper, SAC would also learn a value function in
              addition to the Q-functions. However, this has been found to be
              unnecessary.
            - This version currently implements a fixed entropy regularization
              coefficient, `alpha`. However, an entropy-constrained variant also
              exists, and OpenAI notes that this is preferred by practitioners.
    """
    def __init__(self, env, model, buffer, logger, args):
        super().__init__(env, model, buffer, logger, args)

    """
        Functions to be overloaded from the parent class.
    """

    def _get_action(self, o, deterministic=False):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if self.logger.iteration <= self.args.start_steps:
            a = self.env.action_space.sample()
        else:
            # Select action a
            a = self.model.act(o, deterministic)
        return {'a': a}

    def _update_networks(self):
        # Update networks just like DDPG (but overloading the functions below)
        return DDPG._update_networks(self)

    """
        Algorithm-specific functions.
    """

    ################################ Losses ################################

    def _compute_pi_loss(self, o):
        """
            Compute the loss for the policy.
        """
        pi, logp_pi = self.model.pi(o)
        # Get the smaller of the two Q-values from the Q-networks
        # *not* the target networks
        q_pi = self._clipped_double_q_learning(self.model, o, pi)
        # Entropy-regularized policy loss
        loss_pi = (self.args.alpha * logp_pi - q_pi).mean()
        # TODO: add loss info from pi: pi_info = dict(LogPi=logp_pi.detach().numpy())
        return loss_pi

    ###################### Mean Squared Bellman Error ######################

    def _compute_Bellman_backup(self, o2, r, d):
        """
            Compute the entropy regularized "Bellman backup":
                r + γ max_a' Q_ϕ(s', a') - α log π_θ(a'|s')

                where Q_ϕ(s', a') = min (Q_ϕ_1(s', a'), Q_ϕ_2(s', a'))
        """
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.model.pi(o2)
            # Target Q-values
            q_pi_targ = self._get_Bellman_targets(o2, a2)
            return r + self.args.gamma * (1 - d) * (q_pi_targ - self.args.alpha * logp_a2)

    def _get_Bellman_targets(self, o2, a2):
        """
            Compute targets for the Bellman equation.
        """
        # Use the smaller of the two Q-values to form the targets in the
        # Bellman error
        q_pi_targ = self._clipped_double_q_learning(self.target, o2, a2)
        return q_pi_targ
