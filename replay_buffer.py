import threading
import numpy as np
import torch

class Buffer:
    def __init__(self):
        # Buffer for transitions
        self.obs, self.rewards, self.obs_next, self.dones = [], [], [], []
        # Transitions buffer for outputs from the neural network(s) (e.g. actions, values, etc.)
        self.outputs = {}
        # Buffer for minibatches
        self.mb_obs, self.mb_rewards, self.mb_obs_next, self.mb_dones = [], [], [], []
        # Minibatches buffer for outputs from the neural network(s) (e.g. actions, values, etc.)
        self.mb_outputs = {}

    def store_transition(self, obs, outputs, reward, obs_next, done):
        self.obs.append(obs.copy())
        # Store the various outputs from the network(s)
        for key, value in outputs.items():
            # Create an empty array on the first loop
            if key not in self.outputs:
                self.outputs[key] = []
            self.outputs[key].append(value)
        self.rewards.append(reward)
        self.obs_next.append(obs_next.copy())
        self.dones.append(done)

    def store_episode(self):
        # Need .copy() to allow deletion of the elements from the transition
        # buffers without also deleting them from the minibatch buffers
        self.mb_obs.append(np.asarray(self.obs.copy(), dtype=np.float32))
        # Store the various outputs from the network(s)
        for key in self.outputs.keys():
            # Create an empty array on the first loop
            if key not in self.mb_outputs:
                self.mb_outputs[key] = []
            self.mb_outputs[key].append(np.asarray(self.outputs[key].copy(), dtype=np.float32))
        self.mb_rewards.append(np.asarray(self.rewards.copy(), dtype=np.float32))
        self.mb_obs_next.append(np.asarray(self.obs_next.copy(), dtype=np.float32))
        self.mb_dones.append(np.asarray(self.dones.copy(), dtype=np.bool))

    def load_minibatch(self):
        mb_transitions = {'observations': self.mb_obs,
                          'rewards': self.mb_rewards,
                          'next observations': self.mb_obs_next,
                          'dones': self.mb_dones}
        # Add the various outputs from the network(s) to the minibatch transitions dictionary
        for key, value in self.mb_outputs.items():
            mb_transitions[key] = value
        return mb_transitions

    def _empty_trajectory_buffers(self):
        del self.obs[:]
        for key in self.outputs.keys():
            del self.outputs[key][:]
        del self.rewards[:]
        del self.obs_next[:]
        del self.dones[:]

    def empty_minibatch_buffers(self):
        del self.mb_obs[:]
        for key in self.mb_outputs.keys():
            del self.mb_outputs[key][:]
        del self.mb_rewards[:]
        del self.mb_obs_next[:]
        del self.mb_dones[:]

class PolicyBuffer(Buffer):
    def __init__(self):
        super().__init__()

    def store_episode(self):
        super().store_episode()
        # reset buffers
        super()._empty_trajectory_buffers()

class ActorCriticBuffer(PolicyBuffer):
    def __init__(self):
        super().__init__()
        # Buffer for transitions
        self.values = []
        # Buffer for minibatches
        self.mb_values = []

    def store_transition(self, obs, action, reward, obs_next, done):
        super().store_transition(obs, action, reward, obs_next, done)

    def store_episode(self):
        super().store_episode()
        # Need .copy() to allow deletion of the elements from the transition
        # buffers without also deleting them from the minibatch buffers
        self.mb_values.append(np.asarray(self.values.copy(), dtype=np.float32)) # Final value estimate is made on the final state where no action is taken.
        # reset buffers
        self._empty_trajectory_buffers()

    def store_value(self, value):
        self.values.append(value)

    # def load_minibatch(self):
    #     mb_transitions = super().load_minibatch()
    #     mb_transitions['values'] = self.mb_values
    #     return mb_transitions

    def _empty_trajectory_buffers(self):
        super()._empty_trajectory_buffers()
        del self.values[:]

    def empty_minibatch_buffers(self):
        super().empty_minibatch_buffers()
        del self.mb_values[:]

class ReplayBuffer(Buffer):
    def __init__(self):
        super().__init__()
        # Minibatch arrays
        self.obs, self.actions, self.dones = [], [], []

    def store(self, reward, value, log_prob):
        self.obs.append(obs.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def load(self):
        obs = np.asarray(self.obs, dtype=np.float32)
        actions = np.asarray(self.actions, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.bool)
        return rewards, values, log_probs

    def reset(self):
        del self.obs[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]

class HERBuffer(Buffer):
    def __init__(self, env_params, buffer_size, replay_strategy, replay_k, compute_reward):
        from her_modules.her import her_sampler

        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        # her sampler
        self.her_module = her_sampler(replay_strategy, replay_k, compute_reward)
        self.sample_func = self.her_module.sample_her_transitions
        # create the rollout buffers
        self.ep_obs, self.ep_ag, self.ep_g, self.ep_actions = [], [], [], []
        # create the cycle buffers
        self.mb_obs, self.mb_ag, self.mb_g, self.mb_actions = [], [], [], []
        # create the buffer to store info
        selfs = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
        # thread lock
        self.lock = threading.Lock()

    def _convert_mb_to_arrays(self):
        mb_obs = np.array(self.mb_obs)
        mb_ag = np.array(self.mb_ag)
        mb_g = np.array(self.mb_g)
        mb_actions = np.array(self.mb_actions)
        return mb_obs, mb_ag, mb_g, mb_actions

    def _sample_her(self, mb_obs, mb_ag, mb_g, mb_actions, _preproc_og_func):
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = _preproc_og_func(obs, g)
        return transitions

    def append_rollouts(self, t, obs, ag, g, action):
        if t == 0:
            # reset the rollout buffers
            self.ep_obs, self.ep_ag, self.ep_g, self.ep_actions = [], [], [], []
        self.ep_obs.append(obs.copy())
        self.ep_ag.append(ag.copy())
        self.ep_g.append(g.copy())
        self.ep_actions.append(action.copy())

    def append_final_rollout(self, rollout, obs, ag):
        if rollout == 0:
            # reset the cycle buffers
            self.mb_obs, self.mb_ag, self.mb_g, self.mb_actions = [], [], [], []
        self.ep_obs.append(obs.copy())
        self.ep_ag.append(ag.copy())
        self.mb_obs.append(self.ep_obs)
        self.mb_ag.append(self.ep_ag)
        self.mb_g.append(self.ep_g)
        self.mb_actions.append(self.ep_actions)

    # store the episode
    def store_episode(self, _preproc_og_func):
        mb_obs, mb_ag, mb_g, mb_actions = self._convert_mb_to_arrays()
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            selfs['obs'][idxs] = mb_obs
            selfs['ag'][idxs] = mb_ag
            selfs['g'][idxs] = mb_g
            selfs['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size
        transitions = self._sample_her(mb_obs, mb_ag, mb_g, mb_actions, _preproc_og_func)
        return transitions

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in selfs.keys():
                temp_buffers[key] = selfs[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
