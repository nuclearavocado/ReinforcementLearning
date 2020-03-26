import gym
# Local
import models
import policies
import replay_buffer # TODO: from replay_buffer.policy_buffer import PolicyBuffer
import arguments
import loggers

def policy_gradient(env, env_params):
    from agents import reinforce
    buffer = replay_buffer.PolicyBuffer()#env_params, args.buffer_size)
    model = models.MLP(env_params)
    if env_params['discrete_env']: # Discrete
        policy = policies.CategoricalPolicy
    else: # Continuous
        policy = policies.GaussianPolicy
        args.policy = 'Gaussian'
    logger = loggers.TensorBoardLogger(args)
    agent = reinforce.REINFORCE(args, env, env_params, logger, buffer, policy, model)
    return agent

def actor_critic(env, env_params):
    from agents import actor_critic
    buffer = replay_buffer.ActorCriticBuffer()
    actor = models.Actor(env_params)
    critic = models.Critic(env_params)
    if env_params['discrete_env']: # Discrete
        policy = policies.CategoricalPolicy
    else: # Continuous
        policy = policies.GaussianPolicy
        args.policy = 'Gaussian'
    logger = loggers.TensorBoardLogger(args)
    agent = actor_critic.ActorCritic(args, env, env_params, logger, buffer, policy, actor, critic)
    return agent

def vpg(env, env_params):
    from agents import vpg
    buffer = replay_buffer.ActorCriticBuffer()
    actor = models.Actor(env_params)
    critic = models.Critic(env_params)
    if env_params['discrete_env']: # Discrete
        policy = policies.CategoricalPolicy
    else: # Continuous
        policy = policies.GaussianPolicy
        args.policy = 'Gaussian'
    logger = loggers.TensorBoardLogger(args)
    agent = vpg.VPG(args, env, env_params, logger, buffer, policy, actor, critic)
    return agent

def main(args):
    env = gym.make(args.env)
    # env = gym.make('EURUSD-v0')
    env_params = arguments.get_env_params(env, args)

    if args.algo == 'reinforce':
        ''' Policy Gradient '''
        agent = policy_gradient(env, env_params)
    elif args.algo == 'actor-critic':
        ''' Actor-Critic '''
        agent = actor_critic(env, env_params)
    elif args.algo == 'vpg':
        ''' Vanilla Policy Gradient '''
        agent = vpg(env, env_params)

    agent.train()

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)
