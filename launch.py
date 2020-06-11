import numpy as np
import torch
import gym
from gym.spaces import Box, Discrete, Dict
# Local
from arguments import get_args
from loggers import TensorBoardLogger

"""
    Arguments
"""
# Initialize the default parameters and get and arguments from the console
args = get_args()

"""
    Seed
"""
# If desired, seed NumPy and PyTorch for reproducability
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

"""
    Environment
"""
# Initialize the environment
env = gym.make(args.env)
# If desired, seed the environment for reproducability
if args.seed is not None:
    env.seed(args.seed)
# Get environment parameters
if env.spec.max_episode_steps is not None:
    args.max_ep_len = env.spec.max_episode_steps
else:
    # Default to episode of 200 timesteps in length
    args.max_ep_len = 200

# Observation space
if isinstance(env.observation_space, Discrete):
    args.obs_dim = env.observation_space.n
elif isinstance(env.observation_space, Box):
    args.obs_dim = env.observation_space.shape
elif isinstance(env.observation_space, Dict):
    if isinstance(env.observation_space["observation"], Discrete):
        args.obs_dim = env.observation_space["observation"].n
        args.goal_dim = env.observation_space["desired_goal"].n
    elif isinstance(env.observation_space["observation"], Box):
        args.obs_dim = env.observation_space["observation"].shape[0]
        args.goal_dim = env.observation_space["desired_goal"].shape[0]

# Action space
if isinstance(env.action_space, Discrete):
    args.act_dim = env.action_space.shape
    # Raise an exception if we try to use Q-learning on a discrete environment (TODO: implement DQN, discrete SAC...)
    if args.algo in ["ddpg", "td3", "sac"]:
        raise ValueError(f"The '{args.algo}' algorithm is designed for continuous " +
                         f"environments, but '{args.env}' is discrete.")
elif isinstance(env.action_space, Box):
    args.act_dim = env.action_space.shape[0]
else:
    raise NotImplementedError

"""
    Neural Network
"""
# Create the neural network model
model_kwargs=dict(hidden_sizes=[args.hid]*args.l, algo=args.algo)
if args.algo in args.policy_gradient:
    from models import MLPActorCritic
    model = MLPActorCritic(env.observation_space, env.action_space, **model_kwargs)
elif args.algo in args.q_learning:
    from models import MLPActorCriticQ
    model = MLPActorCriticQ(env.observation_space, env.action_space, **model_kwargs)
else:
    raise ValueError(f"Expected `algo` argument to be one of " +
                     f"{algos}, but got '{args.algo}'.")

"""
    Logger
"""
# Create the logger for printing the reward and displaying TensorBoard graphs
args.reward_threshold = env.spec.reward_threshold
logger = TensorBoardLogger(args)

"""
    Buffer
"""
# Create the memory buffer
buf_kwargs = {"size": args.steps,
              "obs_dim": args.obs_dim,
              "act_dim": args.act_dim}

if args.algo in args.policy_gradient:
    buf_kwargs["gamma"] = args.gamma
    if args.algo == "reinforce":
        from replay_buffer import PolicyGradientBuffer
        buffer = PolicyGradientBuffer(**buf_kwargs)
    else:
        from replay_buffer import ActorCriticBuffer
        buf_kwargs["lam"] = args.lam
        buffer = ActorCriticBuffer(**buf_kwargs)
elif args.algo in args.q_learning:
    if args.buffer == "her":
        raise NotImplementedError(f"Support for {args.algo} is still under " +
                                  f"development.")
    else:
        from replay_buffer import QLearningReplayBuffer
        buffer = QLearningReplayBuffer(**buf_kwargs)
else:
    algos = tuple(args.policy_gradient + args.q_learning)
    raise ValueError(f"Expected `algo` argument to be one of " +
                     f"{algos}, but got '{args.algo}'.")

"""
    Learning algorithm
"""
# Create the agent's learning algorithm
if args.algo in ["dqn", "a2c", "npg", "trpo"]:
    raise NotImplementedError(f"Support for {args.algo} is still under " +
                              f"development.")
elif args.algo in args.policy_gradient:
    if args.algo == "reinforce":
        from agents.reinforce import REINFORCE
        agent = REINFORCE(env, model, buffer, logger, args)
    elif args.algo == "vpg":
        from agents.vpg import VPG
        agent = VPG(env, model, buffer, logger, args)
    elif args.algo == "ppo":
        from agents.ppo import PPO
        agent = PPO(env, model, buffer, logger, args)
elif args.algo in args.q_learning:
    if isinstance(env.action_space, Box):
        # Action limit for clamping
        # Critically: assumes all dimensions share the same bound!
        args.act_limit = env.action_space.high[0]
    if args.algo == "ddpg":
        from agents.ddpg import DDPG
        agent = DDPG(env, model, buffer, logger, args)
    elif args.algo == "td3":
        from agents.td3 import TD3
        agent = TD3(env, model, buffer, logger, args)
    elif args.algo == "sac":
        from agents.sac import SAC
        agent = SAC(env, model, buffer, logger, args)
else:
    algos = tuple(args.policy_gradient + args.q_learning)
    raise NotImplementedError(f"Expected `algo` argument to be one of " +
                              f"{algos}, but got '{args.algo}'.")

"""
    Train
"""
# Train the agent!
agent.train()
