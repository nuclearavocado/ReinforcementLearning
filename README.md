# Deep Learning Library

This is a library of deep learning algorithms and architectures, covering supervised-, unsupervised- and reinforcement-learning.

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
- [x] [PPO](https://arxiv.org/abs/1707.06347) (Proximal Policy Optimization)
- [ ] DQN (Deep Q-Learning Network)
- [ ] DDPG (Deep Deterministic Policy Gradient)
- [ ] [HER](https://arxiv.org/abs/1707.01495) (Hindsight Experience Replay)
- [ ] TD3 (Twin Delayed DDPG)
- [ ] [SAC](https://arxiv.org/abs/1801.01290) (Soft Actor-Critic)

## Update Info
:rocket: **2020-03-21** - Launch!  

## TODO List
#TODO: Complete

## Requirments
#TODO: Complete

## Installation
#TODO: Complete

## Instruction
#TODO: Complete

## Code Structures
#TODO: Complete

## Main design considerations catalyzing the need for a custom library
- To be easy to use (inspired by Keras, scikit-learn etc.)
- To be a learning tool: the underlying code is meant to be read, not abstracted away
- To copy the layout of algorithms as defined in papers

### Ashley Hill's criticisms levelled at OpenAI's Baselines:
- No unified structure for algorithms and undocumented functions and classes
- No custom policies for DDPG; only available from the run script
- Common interface only available via the run script
- Rudimentary logging of training information (no loss nor graph)
- Ipython / Notebook unfriendly
- No custom callbacks except for DQN
- Not PEP8 compliant (subsequently fixed)
- Code build now considered failing
- Latest RL algorithms, e.g. SAC and TD3 are not included
- Only HER support for DDPG

## Acknowledgements
The main library that inspired this work is François Chollet's Keras, followed by the scikit-learn interface.
The reinforcement-learning is based on OpenAI's [Baselines](https://github.com/openai/baselines/), OpenAI's [SpinningUp](https://github.com/openai/spinningup/), and TianHong Dai's implementation of [Hindsight Experience Replay](https://github.com/TianhongDai/hindsight-experience-replay) in PyTorch, itself based on OpenAI's [HER](https://github.com/openai/baselines/tree/master/baselines/her) implementation in TensorFlow. It was also influenced by Ashley Hill's [Stable Baselines](https://github.com/hill-a/stable-baselines).
