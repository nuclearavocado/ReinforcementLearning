# Libraries
import time
import torch
import torch.optim as optim
# Local
from agents import vpg

class PPO(vpg.VPG):

    """
        Algorithm:      PPO (Proximal Policy Optimization)
        Paper:          https://arxiv.org/abs/1707.06347
        Paper authors:  John Schulman, Filip Wolski, Prafulla Dhariwal,
                        Alec Radford, and Oleg Klimov
        Institution:    OpenAI

        Like TRPO, the key question for PPO is how to take the biggest possible
        policy improvement step without causing performance to collapse.

        Surrogate Loss
            The key insight for PPO is that much of the complexity of TRPO can
            be abandoned by computing a measure of how the new policy performs
            relative to the old policy.
    """

    def __init__(self, env, model, buffer, logger, args):
        super().__init__(env, model, buffer, logger, args)
        self.toc = time.time()

    # ----------------------------------------------------------
    # Functions to be overloaded from the parent class

    def _get_action(self, o):
        return super()._get_action(o)

    def _update_networks(self, data):
        super()._update_networks(data)

    # ----------------------------------------------------------
    # Algorithm specific functions

    def _update_policy(self, data):
        # Train policy with one or more steps of gradient descent
        for i in range(self.args.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self._compute_policy_loss(data)
            kl = pi_info['kl']
            max_kl = 1.5 * self.args.target_kl
            if kl > max_kl:
                print(f'Early stopping at step {i} due to reaching a kl {kl:.4f}, greater than max kl {max_kl:.4f}.')
                break
            loss_pi.backward()
            self.pi_optimizer.step()

    def _compute_policy_loss(self, data, action_values='adv'):
        """
            Calculate surrogate loss
        """
        o, a, av, logp_old = data["obs"], \
                             data["act"], \
                             data[action_values], \
                             data["logp"]

        """
            Compute the loss:

            Surrogate advantage (a measure of how policy π_θ performs relative to
            the old policy π_θ_k using data from the old policy):
                L(θ_k,θ) = E_{s,a~π_θ_k}[ π_θ(a,s)/π_θ_k(a,s) * A^{π_θ_k}(s,a) ]

            Probability ratio:
                π_θ(a,s)/π_θ_k(a,s)

            Clipped surrogate advantage:
                clip(π_θ(a,s)/π_θ_k(a,s), 1 - ϵ, 1 + ϵ) * A^{π_θ_k}(s,a)

            Loss:
                L(s,a,θ_k,θ) = min(π_θ(a,s)/π_θ_k(a,s) * A^{π_θ_k}(s,a), clip(π_θ(a,s)/π_θ_k(a,s), 1 - ϵ, 1 + ϵ) * A^{π_θ_k}(s,a))
                             = min(Surrogate advantage, Clipped surrogate advantage)
        """

        # Policy loss
        pi, logp = self.model.pi(o, a)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.args.clip_ratio, 1+self.args.clip_ratio) * av
        loss_pi = -(torch.min(ratio * av, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.args.clip_ratio) | ratio.lt(1-self.args.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info
