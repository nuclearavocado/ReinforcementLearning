# Deep Learning Library

This is a library of deep learning algorithms and architectures, covering supervised-, unsupervised- and reinforcement-learning.

## Guiding principles
There are many deep learning libraries out there, [so why add another one](https://xkcd.com/927/)?

![xkcd standards](https://imgs.xkcd.com/comics/standards.png)

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

## TODO List
#TODO: Complete

## Requirements
- pytorch=1.3.1
- gym=0.14.0

## Installation
#TODO: Complete

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
## Code Structures
#TODO: Complete

## Acknowledgements
The main library that inspired this work is François Chollet's Keras, followed by the scikit-learn interface.
The reinforcement-learning is based on OpenAI's [Baselines](https://github.com/openai/baselines/), OpenAI's [SpinningUp](https://github.com/openai/spinningup/), and TianHong Dai's implementation of [Hindsight Experience Replay](https://github.com/TianhongDai/hindsight-experience-replay) in PyTorch, itself based on OpenAI's [HER](https://github.com/openai/baselines/tree/master/baselines/her) implementation in TensorFlow. It was also influenced by Ashley Hill's [Stable Baselines](https://github.com/hill-a/stable-baselines).
