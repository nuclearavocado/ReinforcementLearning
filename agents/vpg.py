# Libraries
import torch
import torch.optim as optim
# Local
from agents import reinforce

class VPG(reinforce.REINFORCE):

    """
        Algorithm:      VPG with GAE (Vanilla Policy Gradient with Generalized
                        Advantage Estimation)
        Paper:          https://arxiv.org/abs/1506.02438
        Paper authors:  John Schulman, Philipp Moritz, Sergey Levine,
                        Michael I. Jordan and Pieter Abbeel
        Institution:    University of California, Berkeley

        VPG is similar to REINFORCE, except that it adds a second network to
        estimate the value of being in a given state. This two-network approach
        is known as an actor-critic algorithm.
    """

    def __init__(self, env, model, buffer, logger, args):
        super().__init__(env, model, buffer, logger, args)
        self.v_params = self.model.v.parameters()
        self.v_optimizer = optim.Adam(self.v_params, lr=args.v_lr)

    # ----------------------------------------------------------
    # Functions to be overloaded from the parent class

    def _get_action(self, o):
        a, v, logp = self.model.step(o)
        return {'a': a, 'v': v, "logp": logp}

    def _update_networks(self, data):
        super()._update_networks(data)
        v_l_old = self._compute_value_loss(data).item()
        # Perform update
        self._update_value(data)

    # ----------------------------------------------------------
    # Algorithm specific functions

    def _update_policy(self, data):
        # Freeze value-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        self._freeze(self.v_params)
        super()._update_policy(data)
        # Unfreeze value-network so you can optimize it at next step.
        self._unfreeze(self.v_params)

    def _update_value(self, data):
        # Freeze policy-network so you don't waste computational effort
        # computing gradients for it during the value learning step.
        self._freeze(self.pi_params)
        # Train value-network with one or more steps of gradient descent
        for _ in range(self.args.train_v_iters):
            self.v_optimizer.zero_grad()
            loss_v = self._compute_value_loss(data)
            loss_v.backward()
            self.v_optimizer.step()
        # Unfreeze policy-network so you can optimize it at next step.
        self._unfreeze(self.pi_params)

    def _compute_policy_loss(self, data, action_values='adv'):
        """
            For actor-critic, the `action_values` are equal to the advantages:
                q_π(s,a) = A(s,a)
        """
        return super()._compute_policy_loss(data, action_values=action_values)

    def _compute_value_loss(self, data):
        o, r = data["obs"], data['ret']
        return ((self.model.v(o) - r)**2).mean()
