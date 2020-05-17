import gym
# Local
from arguments import get_args
from models import MLPActorCritic
from loggers import TensorBoardLogger

# Arguments
args = get_args()

# Environment
env = gym.make(args.env)

# Model
model_kwargs=dict(hidden_sizes=[args.hid]*args.l)
model = MLPActorCritic(env.observation_space, env.action_space, **model_kwargs)

# Logger
args.reward_threshold = env.spec.reward_threshold
logger = TensorBoardLogger(args)

# Buffer
buf_kwargs = {'size': args.steps,
              'obs_dim': env.observation_space.shape,
              'act_dim': env.action_space.shape,
              'gamma': args.gamma}

policy_gradient = ['reinforce', 'vpg', 'npg', 'trpo', 'ppo']
q_learning = ['ddpg', 'td3', 'sac']

if (args.algo in policy_gradient):
    if args.algo == "reinforce":
        from replay_buffer import PolicyGradientBuffer
        buffer = PolicyGradientBuffer(**buf_kwargs)
    else:
        from replay_buffer import ActorCriticBuffer
        buf_kwargs["lam"] = args.lam
        buffer = ActorCriticBuffer(**buf_kwargs)

# Algorithm
if args.algo == "reinforce":
    from agents.reinforce import REINFORCE
    agent = REINFORCE(env, model, buffer, logger, args)
elif args.algo == "vpg":
    from agents.vpg import VPG
    agent = VPG(env, model, buffer, logger, args)
elif args.algo == "npg":
    from agents.npg import NPG
    agent = NPG(env, model, buffer, logger, args)
elif args.algo == "trpo":
    from agents.trpo import TRPO
    agent = TRPO(env, model, buffer, logger, args)
elif args.algo == "ppo":
    from agents.ppo import PPO
    agent = PPO(env, model, buffer, logger, args)

# Train
agent.train()
