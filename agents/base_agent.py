import numpy as np
import torch

class BaseAgent:

    def __init__(self, env, model, buffer, logger, args):
        self.env = env
        self.model = model
        self.args = args
        self.logger = logger
        self.buffer = buffer
        # If desired, seed the environment for reproducability
        if self.args.seed is not None:
            self.env.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

    # ----------------------------------------------------------
    # This is the only public function for agents

    def train(self):
        for self.logger.epoch in range(self.args.epochs):
            print(f"Epoch {self.logger.epoch}:")
            # Collect a set of trajectories by running the agent in the environment
            self._sample_environment()
            # Log the environment interation
            self.logger.finish_epoch()
            # Use the experience collected to update the networks
            self._update()

    # ----------------------------------------------------------
    # These functions are shared across all agents and do not require
    # modification by the child classes

    #################### Agent/Environment interaction loop ####################

    def _sample_environment(self):
        # Reset environment and observe initial state s_0
        o, ep_len = self.env.reset(), 0
        # Run the agent in the environment until we reach the maximum number of timesteps, or until done
        for t in range(self.args.steps):
            # If desired, render the environment
            if self.args.render:
                self.env.render()
            # Pass state to agent and return outputs
            outputs = self._get_nn_outputs(o)
            # Execute a in the environment and observe next state s', reward r, and done signal d
            o_next, r, d, _ = self.env.step(outputs['action'])
            ep_len += 1
            # Store r in the buffer
            self.buffer.store(o, r, outputs)
            # Log reward
            self.logger.add_scalars('reward', r)
            # Observe state s (critical!)
            o = o_next
            # If s' is terminal, reset the environment state
            timeout = ep_len == self.args.max_ep_len
            terminal = d or timeout
            epoch_ended = t==self.args.steps-1
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory interrupted at %d steps.'%ep_len, flush=True)
                if timeout or epoch_ended:
                    _, v, _ = self.model.step(o)
                else:
                    v = 0
                # Do any necessary computations at the end of the trajectory
                self.buffer.finish_path(v)
                # Reset environment and observe initial state s_0
                o, ep_len = self.env.reset(), 0
                # Increase episode number if terminal iteration
                self.logger.episode += 1
            # Increase iteration number every timestep
            self.logger.iteration += 1

    #################### Agent update ####################

    def _update(self):
        # Load the transitions from the buffer
        data = self.buffer.get()
        self._update_networks(data)

    # ----------------------------------------------------------
    # These functions are called by _sample_environment() and _update() and
    # should be overloaded in the child classes

    '''
        _sample_environment()
    '''

    def _get_nn_outputs(self, o):
        raise NotImplementedError

    '''
        _update()
    '''

    def _update_networks(self, data):
        raise NotImplementedError
