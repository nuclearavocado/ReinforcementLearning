# Libraries
import torch
import torch.optim as optim
# Local
from agents import reinforce

'''
    Algorithm:      VPG with GAE (Vanilla Policy Gradient with Generalized
                    Advantage Estimation)
    Paper:          https://arxiv.org/abs/1506.02438
    Paper authors:  John Schulman, Philipp Moritz, Sergey Levine,
                    Michael I. Jordan and Pieter Abbeel
    Institution:    University of California, Berkeley
'''

class VPG(reinforce.REINFORCE):

    def __init__(self, env, model, buffer, logger, args):
        super().__init__(env, model, buffer, logger, args)
        self.vf_optimizer = optim.Adam(self.model.v.parameters(), lr=args.vf_lr)

    # ----------------------------------------------------------
    # Functions to be overloaded from the parent class

    def _get_nn_outputs(self, o):
        a, v, logp = self.model.step(o)
        output = {'action': a, 'value': v, 'log probability': logp}
        return output

    def _update_networks(self, data):
        super()._update_networks(data)
        v_l_old = self._compute_critic_loss(data).item()
        self._update_critic(data)

    # ----------------------------------------------------------
    # Algorithm specific functions

    def _update_actor(self, data):
        super()._update_actor(data)

    def _update_critic(self, data):
        for _ in range(self.args.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self._compute_critic_loss(data)
            loss_v.backward()
            self.vf_optimizer.step()

    def _compute_actor_loss(self, data, action_values='adv'):
        '''
            For actor-critic, the `action_values` are equal to the advantages:
                q_π(s,a) = A(s,a)
        '''
        return super()._compute_actor_loss(data, action_values=action_values)

    def _compute_critic_loss(self, data):
        o, r = data['obs'], data['ret']
        return ((self.model.v(o) - r)**2).mean()
