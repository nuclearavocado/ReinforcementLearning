import numpy as np
import torch

class BaseAgent:
    def __init__(self, args, env, env_params, logger, buffer, policy):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.logger = logger
        self.buffer = buffer
        self.policy = policy
        self.iteration = 0
        # If desired, seed the environment for reproducability
        if self.args.seed is not None:
            self.env.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

    # ----------------------------------------------------------
    # This is the only public function for agents

    def train(self):
        for self.episode in range(self.args.n_epochs):
            # Collect a set of trajectories by running the agent in the environment
            self._sample_environment()
            # Use the experience collected to update the networks
            self._update()

    # ----------------------------------------------------------
    # These functions are shared across all agents

    def _sample_environment(self):
        # Reset memory buffers
        self.buffer.empty_minibatch_buffers()
        # Sample the environment for `n_batches`
        for self.batch in range(self.args.n_batches):
            # Reset environment and observe initial state s_0
            obs_next = self.env.reset()
            # Run the agent in the environment until we reach the maximum number of timesteps, or until done
            for t in range(self.env_params['max_timesteps']):
                # If desired, render the environment
                if self.args.render:
                    self.env.render()
                # Observe state s
                obs = obs_next
                # Pass state to agent and return outputs
                outputs = self._get_nn_outputs(obs)
                # Execute a in the environment and observe next state s',
                # reward r, and done signal d
                obs_next, reward, done, _ = self.env.step(outputs['actions'])
                # Store r in the buffer
                self.buffer.store_transition(obs, outputs, reward, obs_next, done)
                # If s' is terminal, reset the environment state
                if done:
                    break
            # Do any necessary computations at the end of the trajectory
            self._finish_trajectory()
            # Store logged variables
            self._logs()
            # Print logs to the command line/to TensorBoard
            self.logger.dump_logs(self.iteration, self.args.log_interval, reward_threshold=self.env.spec.reward_threshold)
            # Store the episode in the buffer.
            # (N.B. this must follow logging, as the rewards will be deleted
            # from the buffer when store_episode() is called)
            self.buffer.store_episode()
            # Increment the iteration number
            self.iteration += 1

    def _update(self):
        # Load the last minibatch of transitions from the buffer
        mb = self.buffer.load_minibatch()
        # Preprocess the minibatch
        mb = self._process_trajectories(mb)
        # Concatenate minibatch trajectories and convert to tensors
        for key, value in mb.items():
            value = np.concatenate([x for x in value])
            mb[key] = torch.from_numpy(value).float()
        # Update networks
        self._update_networks(mb)

    # ----------------------------------------------------------
    # These functions are called by _sample_environment() and _update() and
    # should be overloaded in the child classes

    def _get_nn_outputs(self, obs):
        raise NotImplementedError

    def _finish_trajectory(self):
        pass

    def _logs(self):
        # Store the episode reward to the logger
        self.logger.add_scalar('episode reward', sum(self.buffer.rewards))

    def _process_trajectories(self, mb):
        raise NotImplementedError

    def _update_networks(self, mb, retain_graph=False):
        raise NotImplementedError
