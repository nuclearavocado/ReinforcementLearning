import numpy as np

# TODO: Unify BasePolicyGradientAgent with BaseAgent
class BaseAgent:
    """
        The properties of BaseAgent are shared by all of the implemented
        reinforcement learning algorithms. This draws attention to what is
        common, and what changes between each algorithm.
    """
    def __init__(self, buffer, logger, args):
        self.args = args
        self.logger = logger
        self.buffer = buffer

    """
        The following functions are shared across all agents and do not need to
        be overloaded in child classes.
    """

    ############################### Training ###############################

    def train(self):
        # This is the only public function for agents
        for self.logger.epoch in range(self.args.epochs):
            print(f"Epoch {self.logger.epoch}: ", end='')
            # Collect a set of trajectories by running the agent in the environment
            self._sample_trajectory()
            # TODO: Save model
        # Close the window if rendering and finished training
        if self.args.render:
            self.env.close()

    def _sample_trajectory(self):
        # Reset environment and observe initial state s_0
        o, self.ep_len = self.env.reset(), 0
        # Run the agent in the environment until until done, or until we reach
        # the maximum number of timesteps for a single episode (determined by
        # the environment).
        for self.step in range(self.args.steps):
            # If desired, render the environment
            if self.args.render:
                self.env.render()
            # Sample one step of the environment
            o = self._sample_environment(o)
            # Increase iteration number every timestep
            self.logger.iteration += 1
        # Log the environment interation
        self.logger.finish_epoch()
        # Perform any required computations after collecting an episode
        # trajectory.
        self._finish_trajectory()

    #################### Agent/Environment interaction loop ####################

    def _sample_environment(self, o):
        # Pass state to agent and return dictionary of outputs, including action `a`
        outputs = self._get_action(o)
        # Execute `a` in the environment and observe next observation `o_next`, reward `r`, and done signal `d`
        o_next, r, d, _ = self.env.step(outputs['a'])
        self.ep_len += 1
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        timeout = (self.ep_len + 1 == self.args.max_ep_len)
        d = False if timeout else d
        # Store (o, a, r, o_next, d) in the buffer
        self.buffer.store(o, r, o_next, d, outputs) # self.buffer.store(o, r, outputs)
        # Log reward
        self.logger.add_scalars('reward', r)
        # Determine if the episode is over
        terminal = (d or timeout)
        epoch_ended = (self.ep_len == (self.args.steps - 1))
        # Perform any required computations after one step of interacting
        # with the environment.
        self._finish_step(terminal, timeout, epoch_ended)
        if not (terminal or epoch_ended):
            # Observe state s'
            return o_next
        else:
            # Increase episode number if terminal iteration
            self.logger.episode += 1
            # If s' is terminal, reset the environment and observe initial state s_0
            o, self.ep_len = self.env.reset(), 0
            return o

    ################################# Utils ################################

    def _freeze(self, network_parameters):
        # Freeze given network parameters so you don't waste computational
        # effort computing unnecessary gradients.
        for p in network_parameters:
            p.requires_grad = False

    def _unfreeze(self, network_parameters):
        # Unfreeze given network parameters so you can optimize them.
        for p in network_parameters:
            p.requires_grad = True

    def _gradient_descent(self, optimizer, loss_function, data):
        optimizer.zero_grad()
        loss = loss_function(data)
        loss.backward()
        optimizer.step()

    """
        The following functions must be overloaded in child classes.
    """

    def _get_action(self, o):
        raise NotImplementedError("The function _get_action() is called by " +
                                  "_sample_environment() and should be " +
                                  "overloaded in child classes. You are " +
                                  f"calling {self.__class__.__name__} " +
                                  "without having overloaded " +
                                  "_sample_environment().")

class BasePolicyGradientAgent(BaseAgent):
    """
        The properties of BasePolicyGradientAgent are shared by all of the
        implemented policy gradient algorithms. This draws attention to what is
        common, and what changes between each algorithm.
    """
    def __init__(self, env, model, buffer, logger, args):
        super().__init__(buffer, logger, args)
        self.env = env
        self.model = model

    # ----------------------------------------------------------
    # These functions are shared across all agents and do not require
    # modification by the child classes

    def _finish_step(self, terminal, timeout, epoch_ended):
        if epoch_ended and not(terminal):
            print(f'Warning: trajectory interrupted at {self.step} steps.', flush=True)
        if timeout or epoch_ended:
            _, v, _ = self.model.step(o)
        else:
            v = 0
        # Calculate rewards-to-go and/or advantages
        self.buffer.finish_path(v)

    def _finish_trajectory(self):
        # Load the transitions from the buffer
        data = self.buffer.get()
        # Use the experience collected to update the networks
        self._update_networks(data)

    #################### Agent/Environment interaction loop ####################

    def _sample_environment(self, o):
        # Pass state to agent and return dictionary of outputs, including action `a`
        outputs = self._get_action(o)
        # Execute `a` in the environment and observe next observation `o_next`, reward `r`, and done signal `d`
        o_next, r, d, _ = self.env.step(outputs['a'])
        self.ep_len += 1
        # Store r in the buffer
        self.buffer.store(o, r, outputs)
        # Log reward
        self.logger.add_scalars('reward', r)
        # If s' is terminal, reset the environment state
        timeout = (self.ep_len == self.args.max_ep_len)
        terminal = (d or timeout)
        epoch_ended = (self.step == (self.args.steps - 1))
        if terminal or epoch_ended:
            if epoch_ended and not(terminal):
                print(f'Warning: trajectory interrupted at {self.ep_len} steps.', flush=True)
            if timeout or epoch_ended:
                _, v, _ = self.model.step(o)
            else:
                v = 0
            # Do any necessary computations at the end of the trajectory
            self.buffer.finish_path(v)
            # Increase episode number if terminal iteration
            self.logger.episode += 1
            # If s' is terminal, reset the environment and observe initial state s_0
            self.ep_len = 0
            return self.env.reset()
        else:
            # Observe state s'
            return o_next

    #################### Agent update ####################

    # ----------------------------------------------------------
    # These functions should be overloaded in the child class

    def _get_action(self, o):
        # This function is called by _sample_environment()
        # should be overloaded in the child class
        raise NotImplementedError

    def _update_networks(self, data):
        raise NotImplementedError("The function _update_networks() is called " +
                                  "by _finish_trajectory() and should be " +
                                  "overloaded in child classes. You are " +
                                  f"calling {self.__class__.__name__} " +
                                  "without having overloaded _update_networks().")


class BaseQLearningAgent(BaseAgent):
    """
        The properties of BaseQLearningAgent are shared by all of the
        implemented Q-learning algorithms. The purpose of this class is to hide
        away as many of the framework specific software engineering details as
        possible.
    """
    def __init__(self, env, buffer, logger, args):
        super().__init__(buffer, logger, args)
        self.env, self.test_env = env, env

    """
        The following functions are shared across all Q-learning agents and
        do not need to be overloaded in child classes.
    """

    def _finish_step(self, terminal, timeout, epoch_ended):
        """
            Update the networks.
        """
        # Only update after a certain number of iteratins have elapsed, then
        # update every `train_q_iters` iterations.
        if self.logger.iteration >= self.args.update_after:
            if self.logger.iteration % self.args.train_q_iters == 0:
                # Use the experience collected to update the networks
                self._update_networks()

    def _finish_trajectory(self):
        """
            Test the performance of a deterministic version of the agent.
        """
        # Set action noise to zero to take deterministic actions at test time
        self.noise_scale = 0
        for _ in range(self.args.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.args.max_ep_len)):
                # If desired, render the environment
                if self.args.render and self.logger.epoch % self.args.render_interval == 0:
                    self.env.render()
                o, r, d, _ = self.test_env.step(self._get_action(o)['a'])
                # Log reward
                self.logger.add_scalars('test reward', r)
                ep_len += 1
        # Reset the action noise back to its original value
        self.noise_scale = self.args.act_noise
        # Close the render window if not rendering every epoch
        if self.args.render and self.args.render_interval > 1:
            self.env.close()

    """
        The following functions must be overloaded in child classes.
    """

    def _update_networks(self, data):
        raise NotImplementedError("The function _update_networks() is called " +
                                  "by _finish_step() and should be " +
                                  "overloaded in child classes. You are " +
                                  f"calling {self.__class__.__name__} " +
                                  "without having overloaded _update_networks().")
