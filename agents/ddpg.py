"acts"# Libraries
import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
# Local
from agents.base_agents import BaseQLearningAgent

class DDPG(BaseQLearningAgent):

    """
        Algorithm:      DDPG (Deep Deterministic Policy Gradient)
        Paper:          https://arxiv.org/abs/1509.02971
        Paper authors:  Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel,
                        Nicolas Heess, Tom Erez, Yuval Tassa, David Silver,
                        and Daan Wierstra
        Institution:    Google Deepmind

        DDPG implements two main tricks:

        Replay Buffers
             This is a standard feature of all Q-learning algorithms.
             The replay buffer stores previous experiences up to some limit.
             Defining this limit may require experimentation. The buffer should
             be large enough to contain a wide range of experiences and prevent
             overfitting to recent data, but small enough that you don't slow
             down learning.

        Target Networks
            Q-learning algorithms make use of target networks. The term:
                r + γ max_a' Q_ϕ(s', a')
            is called the target, because when we minimize the MSBE loss, we are
            trying to make the Q-function be more like this target.
            Problematically, the target depends on the same parameters we are
            trying to train, ϕ, which makes MSBE minimization unstable. The
            solution is to use a set of parameters which are similar to ϕ, but
            with a time delay, i.e. a second network, called the target network.
            In DDPG, the target network is updated once per main-network update
            by polyak averaging:
                ϕ_targ <– ρ ϕ_targ + (1 - ρ)ϕ
                where ρ is in the closed interval [0, 1]
    """

    def __init__(self, env, model, buffer, logger, args):
        super().__init__(env, buffer, logger, args)
        """ Target networks """
        self.model, self.target = model, deepcopy(model)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        self._freeze(self.target.parameters())
        # Create optimizers
        self.pi_params = self.model.pi.parameters()
        self.pi_optimizer = optim.Adam(self.pi_params, lr=args.pi_lr)
        # Handle inheritance for algorithms with multiple Q-functions
        if args.algo == "ddpg":
            self.q_params = self.model.q.parameters()
            self.q_optimizer = optim.Adam(self.q_params, lr=args.q_lr)
        # This variable switches between the value of act_noise and 0, and
        # allows the DDPG agent to take non-deterministic actions during
        # training, but deterministic actions at test time.
        self.noise_scale = self.args.act_noise

    """
        Functions to be overloaded from the parent class.
    """

    def _get_action(self, o):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration.
        if self.logger.iteration <= self.args.start_steps:
            # Uniformly sample an action
            a = self.env.action_space.sample()
        else:
            # Get action a from policy network
            a = self.model.act(o)
            # Add some noise to the action
            a += self.noise_scale * np.random.randn(self.args.act_dim)
            # Ensure the action is within the limits
            a = np.clip(a, -self.args.act_limit, self.args.act_limit)
        return {'a': a}

    def _update_networks(self):
        # Update `train_q_iters` times
        for self.train_q_iter in range(self.args.train_q_iters):
            # Load the transitions from the buffer
            batch = self.buffer.sample_batch(self.args.batch_size)
            # First run one gradient descent step for Q
            self._update_q(batch)
            # Next run one gradient descent step for pi
            self._update_policy(batch)
            # Finally, update target networks
            self._update_target()

    """
        Algorithm-specific functions.
    """

    ############################### Updating ###############################

    def _update_q(self, data):
        inputs = (data["obs"], data["acts"], data["rews"], data["obs2"], data["dones"])
        # Run one gradient descent step for Q
        self._gradient_descent(self.q_optimizer, self._compute_q_loss, inputs)

    def _update_policy(self, data):
        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        self._freeze(self.q_params)
        # Run one gradient descent step for pi
        self._gradient_descent(self.pi_optimizer, self._compute_pi_loss, data["obs"])
        # Unfreeze value-network so you can optimize it at next step.
        self._unfreeze(self.q_params)

    def _update_target(self):
        """ Update target networks by polyak averaging. """
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.target.parameters()):
                # NB: We use in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.args.polyak)
                p_targ.data.add_((1 - self.args.polyak) * p.data)

    ################################ Losses ################################

    def _compute_q_loss(self, data):
        """
            Compute the loss for the Q-network.
        """
        o, a, r, o2, d = data
        # Get Q-values from the Q-network(s)
        q = self._get_q_values(o, a)
        # Compute Mean Squared Bellman Error (MSBE)
        loss_q = self._compute_MSBE(o2, r, d, q)
        # TODO: add loss info from q(s): loss_info = dict(QVals=q.detach().numpy())
        return loss_q

    def _compute_pi_loss(self, o):
        """
            Compute the loss for the policy.
        """
        pi = self.model.pi(o)
        q_pi = self.model.q(o, pi)
        return -q_pi.mean()

    def _get_q_values(self, o, a):
        """
            Get action-value estimates (Q-values) from network.
        """
        return self.model.q(o, a)

    ###################### Mean Squared Bellman Error ######################

    def _compute_MSBE(self, o2, r, d, q):
        """
            Compute the Mean Squared Bellman Error:
                E[(Q(s, a) - r + γ max_a' Q_ϕ(s', a'))**2]
        """
        # Compute Bellman backup for Q function
        backup = self._compute_Bellman_backup(o2, r, d)
        # MSE loss(es) against Bellman backup
        loss_q = self._compute_MSE(q, backup)
        return loss_q

    def _compute_Bellman_backup(self, o2, r, d):
        """
            Compute the "Bellman backup":
                r + γ max_a' Q_ϕ(s', a')
        """
        with torch.no_grad():
            pi_targ = self.target.pi(o2)
            q_pi_targ = self._get_Bellman_targets(o2, pi_targ)
            return r + self.args.gamma * (1 - d) * q_pi_targ

    def _get_Bellman_targets(self, o2, pi_targ):
        """
            Compute targets for the Bellman equation.
        """
        q_pi_targ = self.target.q(o2, pi_targ)
        return q_pi_targ

    def _compute_MSE(self, q, backup):
        """
            Compute mean squared error between the Q-values from the network and
            the computed Bellman backup.
        """
        return ((q - backup)**2).mean()
