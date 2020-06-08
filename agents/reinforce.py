# Libraries
import torch
import torch.optim as optim
# Local
from agents.base_agents import BasePolicyGradientAgent

class REINFORCE(BasePolicyGradientAgent):

    """
        Algorithm:      REINFORCE (REward Increment = Nonnegative Factor times
                        Offset Reinforcement times Characteristic Eligibility)
        Paper:          https://doi.org/10.1007/BF00992696
        Paper author:   Ronald J. Williams
        Institution:    Northeastern University

        REINFORCE is based on one key idea:

        Policy Gradients
            The key idea behind policy gradients algorithms is to increase the
            probability of taking actions that lead to higher return, and
            decrease the probability of taking actions that lead to lower
            return.
            This is done by taking some actions in the environment, and from
            this, computing a sample estimate of the gradient of the reward
            function.
    """

    def __init__(self, env, model, buffer, logger, args):
        super().__init__(env, model, buffer, logger, args)
        self.pi_params = self.model.pi.parameters()
        self.pi_optimizer = optim.Adam(self.pi_params, lr=args.pi_lr)

    # ----------------------------------------------------------
    # Functions to be overloaded from the parent class

    def _get_action(self, o):
        # Select action `a`, compute value `v`, and log probability `logp`
        a, v, logp = self.model.step(o)
        return {'a': a, 'v': v, "logp": logp}

    def _update_networks(self, data):
        # Get loss and info values before update
        pi_l_old, pi_info_old = self._compute_policy_loss(data)
        pi_l_old = pi_l_old.item()
        # TODO: store pi_l_old and pi_info_old to logger; currently not used
        # Perform update
        self._update_policy(data)

    # ----------------------------------------------------------
    # Algorithm specific functions

    def _update_policy(self, data):
        # Train policy with one or more steps of gradient descent
        for _ in range(self.args.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self._compute_policy_loss(data)
            # TODO: store data from pi_info to logger; currently not used except in PPO
            loss_pi.backward()
            self.pi_optimizer.step()

    def _compute_policy_loss(self, data, action_values='ret'):
        """
            Calculate the policy gradient.
        """
        """

            Args:
            `data`:        dictionary of agent interactions with the
                             environment, including observations, actions,
                             returns/advantages/etc.
            `action_values`: a string defining the key in the input dictionary
                             for the action-value estimate, the choice of which
                             is algorithm specific.
                             See the paper on Generalized Advantage Estimation
                             (GAE) for more details on the different choices of
                             `action_values`:
                             https://arxiv.org/pdf/1506.02438.pdf
        """
        """
            For REINFORCE, the `action_values` are equal to the returns:
                q_π(s,a) = r(s,a)
        """

        o, a, av, logp_old = data["obs"], \
                             data["act"], \
                             data[action_values], \
                             data["logp"]

        """
            Compute the loss:

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
        """

        # Policy loss
        pi, logp = self.model.pi(o, a)
        loss_pi = -(logp * av).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info
