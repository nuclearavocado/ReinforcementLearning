# Deep Learning Library

This is a library of deep learning algorithms and architectures, covering supervised-, unsupervised- and reinforcement-learning.

## Guiding principles
![xkcd standards](https://imgs.xkcd.com/comics/standards.png)

There are many deep learning libraries out there, so why add another one?
This library originates specifically from frustrations encountered with reinforcement-learning library implementations, including [OpenAI's Baselines](https://github.com/openai/baselines/), [OpenAI's SpinningUp](https://github.com/openai/spinningup/), and [Ashley Hill's Stable Baselines](https://github.com/hill-a/stable-baselines).

[Keras](https://keras.io) set the bar for how a real deep-learning library should be built, and this is an attempt to continue in that tradition.

Inspired by [Keras](https://keras.io), this library is designed for human beings. Specifically: human beings as dumb as me.
There are two goals:
1. Enable fast experimentation (something achieved by [Keras](https://keras.io))
2. Didactic (something achieved by [Spinning Up](https://spinningup.openai.com/))

Features in service of the first goal: 
- User friendly. There should be a simple sequence of API calls that minimize the number of user actions required for common use cases.
- Modular. The library should be built around modules that can be easily be added, subtracted and rearranged. For example, transformers and GANs should be standalone modules that can be added to a reinforcement-learning model architecture with a single line of code.
- Extensible. New modules should be simple to add, and have a consistent API, with inheritance of functionality from their algorithmic predecessors.

Features in service of the second goal:
- Algorithmic clarity. The code should be as close to the pseudocode found in the paper as possible, such that it is readable enough to learn from. The backend should not interrupt the flow of the algorithm (I don't want random MPI calls mixed up with a new RL algorithm I'm trying to learn). Conversely, key parts of the algorithm itself should not be abstracted away for ease of use when calling from the terminal, etc. (A combination of these two problems is the primary failure mode of all reinforcement-learning libraries, except for [Spinning Up](https://spinningup.openai.com/))
- Inheritance. This allows the reader can see the "family-tree" of the algorithm (e.g. REINFORCE -> VPG -> NPG -> TRPO), and determine the required prerequisite knowledge from the simpler algorithms before tackling the more complex ones.

## Supervised Learning Architectures
- [ ] [Inception]()
- [ ] [Transformer](https://arxiv.org/abs/1706.03762)

## Unsupervised Learning Architectures
- [ ] [GANs](https://arxiv.org/abs/1406.2661) (Generative Adversarial Networks)

## Reinforcement Learning Algorithms
- [x] [REINFORCE](https://doi.org/10.1007/BF00992696) (REward Increment = Nonnegative Factor x Offset Reinforcement x Characteristic Eligibility)
- [x] VPG (Vanilla Policy Gradient)
- [ ] A2C (Advantage Actor-Critic)
- [ ] [NPG](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf) (Natural Policy Gradient)
- [ ] [TRPO](https://arxiv.org/abs/1502.05477) (Trust Region Policy Optimization)
- [ ] [PPO](https://arxiv.org/abs/1707.06347) (Proximal Policy Optimization)
- [ ] [DQN](https://arxiv.org/abs/1312.5602) (Deep Q-Learning Network)
- [ ] [DDPG](https://arxiv.org/abs/1509.02971) (Deep Deterministic Policy Gradient)
- [ ] [HER](https://arxiv.org/abs/1707.01495) (Hindsight Experience Replay)
- [ ] [TD3](https://arxiv.org/abs/1802.09477) (Twin Delayed DDPG)
- [ ] [SAC](https://arxiv.org/abs/1801.01290) (Soft Actor-Critic)

## Update Info
:rocket: **2020-03-21** - Launch!

:triangular_flag_on_post: **2020-04-17** - Overhaul to bring in line with SpinningUp PyTorch release

## Requirements
- pytorch=1.3.1
- gym=0.14.0

## Installation
Download and unzip the repository, or run:
```
git clone git@github.com:nuclearavocado/deep-learning.git
```

## Instructions
To train an agent, run:
```
python launch.py --<arguments>
```
For example, to train a VPG agent on Acrobot-v1 and render it:
```
python launch.py --algo vpg --env Acrobot-v1 --render
```
(The default arguments train a REINFORCE agent on CartPole-v1.)

To visualise training, this library uses Tensorboard.
```
tensorboard --logdir=./graphs
```

## Code Structure
The main file is _launch.py_. This collects the various elements required to run a reinforcement learning agent, as defined by the input arguments. This includes the environment (e.g. CartPole-v1), type of agent algorithm (e.g. REINFORCE, DQN, etc.), type of replay buffer (e.g. HER), type of policy (e.g. Gaussian/Categorical), and sets up a logger for printing the reward and saving data for graphs in TensorBoard etc.

_launch.py_ then runs _agent.train()_.

The _train()_ function is located in _base_agent.py_, the class from which all RL-algorithms inherit. It consists of an environment sampling step, followed by a network update step:
```
def train(self):
        for self.episode in range(self.args.n_epochs):
            # Collect a set of trajectories by running the agent in the environment
            self._sample_environment()
            # Use the experience collected to update the networks
            self._update()
```
The sample environment step should be familiar, as all RL-algorithms follow this fundamental structure:
```
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
```
This is followed by the update step:
```
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
```
Each RL-algorithm then has its own way in which it performs _\_get_nn_outputs()_, _\_process_trajectories()_, and _\_update_networks()_, which can be found in the _\<Agent\>_ class in the corresponding _\<agent\>.py_ file.

## Acknowledgements
The main library that inspired this work is François Chollet's Keras, followed by the scikit-learn interface.
The reinforcement-learning is based on OpenAI's [Baselines](https://github.com/openai/baselines/), OpenAI's [SpinningUp](https://github.com/openai/spinningup/), and TianHong Dai's implementation of [Hindsight Experience Replay](https://github.com/TianhongDai/hindsight-experience-replay) in PyTorch, itself based on OpenAI's [HER](https://github.com/openai/baselines/tree/master/baselines/her) implementation in TensorFlow. It was also influenced by Ashley Hill's [Stable Baselines](https://github.com/hill-a/stable-baselines).
