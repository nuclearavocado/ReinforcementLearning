import numpy as np
import torch
import utils
import scipy.signal

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class BaseBuffer:
    """
    A basic buffer for storing trajectories experienced by an agent interacting
    with the environment.
    """

    def __init__(self, size=4000, obs_dim=None, act_dim=None, gamma=0.99):
        self.obs_buf = np.zeros(utils.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, o, r, outputs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size # buffer has to have room so you can store
        self._store_transition(o, r, outputs)
        self.ptr += 1

    def _store_transition(self, o, r, outputs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        o, r, a = o, r, \
                  outputs['action']
        self.obs_buf[self.ptr] = o
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back over the buffer to where the
        trajectory started, and performs various calculations on the to use as
        targets for the neural networks.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        self._process_trajectory(path_slice, last_val)
        self.path_start_idx = self.ptr

    def _process_trajectory(self, path_slice, last_val):
        raise NotImplementedError

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer. Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf)
        data = self._add_data(data)
        # Convert to dictionary of tensors
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

    def _add_data(self, data):
        raise NotImplementedError

class PolicyGradientBuffer(BaseBuffer):
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, size=4000, obs_dim=None, act_dim=None, gamma=0.99):
        super().__init__(size=size, obs_dim=obs_dim, act_dim=act_dim, gamma=gamma)
        self.logp_buf = np.zeros(size, dtype=np.float32)

    def store(self, o, r, outputs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        super().store(o, r, outputs)

    def _store_transition(self, o, r, outputs):
        super()._store_transition(o, r, outputs)
        logp = outputs['log probability']
        self.logp_buf[self.ptr] = logp

    def finish_path(self, last_val=0):
        super().finish_path(last_val)

    def _process_trajectory(self, path_slice, last_val):
        """
        Computes the rewards-to-go for each state.
        """
        rews = np.append(self.rew_buf[path_slice], last_val)
        self._rewards_to_go(path_slice, rews)

    def _rewards_to_go(self, path_slice, rews):
        # the next line computes rewards-to-go
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

    def get(self):
        return super().get()

    def _add_data(self, data):
        data["logp"] = self.logp_buf
        return data

class ActorCriticBuffer(PolicyGradientBuffer):
    """
    A buffer for storing trajectories experienced by an Actor Critic agent
    interacting with the environment, and using Generalized Advantage Estimation
    (GAE-Lambda) for calculating the advantages of state-action pairs.
    """

    def __init__(self, size=4000, obs_dim=None, act_dim=None, gamma=0.99, lam=0.95):
        super().__init__(size=size, obs_dim=obs_dim, act_dim=act_dim, gamma=gamma)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.lam = lam

    def store(self, o, r, outputs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        super().store(o, r, outputs)

    def _store_transition(self, o, r, outputs):
        super()._store_transition(o, r, outputs)
        v = outputs['value']
        self.val_buf[self.ptr] = v

    def finish_path(self, last_val=0):
        super().finish_path(last_val)

    def _process_trajectory(self, path_slice, last_val):
        """
        Computes advantage estimates with GAE-Lambda, and the rewards-to-go for
        each state, to use as the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        self._GAE_Lambda(path_slice, rews, vals)
        self._rewards_to_go(path_slice, rews)

    def _GAE_Lambda(self, path_slice, rews, vals):
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

    def _rewards_to_go(self, path_slice, rews):
        super()._rewards_to_go(path_slice, rews)

    def get(self):
        return super().get()

    def _add_data(self, data):
        super()._add_data(data)
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data["adv"] = self.adv_buf
        return data
