'''
    Algorithm:      VPG with GAE (Vanilla Policy Gradient with Generalized
                    Advantage Estimation)
    Paper:          https://arxiv.org/abs/1506.02438
    Paper authors:  John Schulman, Philipp Moritz, Sergey Levine,
                    Michael I. Jordan and Pieter Abbeel
    Institution:    University of California, Berkeley
'''

# Local
import action_value_functions
from agents.actor_critic import ActorCritic

class VPG(ActorCritic):
    def __init__(self, args, env, env_params, logger, buffer, policy, actor=None, critic=None):
        super().__init__(args, env, env_params, logger, buffer, policy, actor, critic)

    # ----------------------------------------------------------
    # Functions to be overloaded from the parent class

    def _get_nn_outputs(self, obs):
        return super()._get_nn_outputs(obs)

    def _finish_trajectory(self):
        super()._finish_trajectory()

    def _logs(self):
        super()._logs()

    def _process_trajectories(self, mb):
        return super()._process_trajectories(mb)

    def _update_networks(self, mb, retain_graph=False):
        super()._update_networks(mb, retain_graph=retain_graph)

    # ----------------------------------------------------------
    # Algorithm specific functions

    def _compute_advantages(self, rewards, values):
        '''
            Generalized Advantage Estimation
        '''
        advantages = action_value_functions.GAE_Lambda(rewards, values, self.args.gamma, self.args.lam)
        return advantages
