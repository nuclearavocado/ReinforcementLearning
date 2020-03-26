import argparse
import gym

def get_args():
    parser = argparse.ArgumentParser(description='A description of the repo')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='the environment name (default: CartPole-v1)')
    parser.add_argument('--algo', type=str, default='reinforce',
                        help='which RL algorithm to use (default: reinforce)')
    parser.add_argument('--buffer', type=str, default='ReplayBuffer',
                        help='which type of buffer to use (default: ReplayBuffer)')
    parser.add_argument('--model', type=str, default='mlp',
                        help='which RL algorithm to use (default: mlp)')
    parser.add_argument('--policy', type=str, default='Categorical',
                        help='which RL algorithm to use (default: Categorical)')
    parser.add_argument('--n-epochs', type=int, default=400,
                        help='the number of epochs to train the agent (default: 1000)')
    parser.add_argument('--n-batches', type=int, default=5,
                        help='the number of times to cycle through the policy before performing a gradient update (default: 5)')
    parser.add_argument('--n-pi-updates', type=int, default=1,
                        help='the number of times to update the actor network after a batch of episodes (default: 1)')
    parser.add_argument('--n-v-updates', type=int, default=1,
                        help='the number of times to update the critic network after a batch of episodes (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='Lambda for Generalized Advantage Estimation (default: 0.95)')
    parser.add_argument('--seed', type=int, default=69, metavar='N',
                        help='random seed (default: 69)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()
    return args

def get_env_params(env, args):
    obs = env.reset()
    # close the environment
    env_params = {'observation_space': env.observation_space,
             'action_space': env.action_space, # alternative gets class name: env.action_space.__class__.__name__
             'max_timesteps': env._max_episode_steps,
             }

    # Action space
    discrete_env = isinstance(env.action_space, gym.spaces.Discrete)
    env_params['discrete_env'] = discrete_env
    if discrete_env: # Discrete
        env_params['actions'] = env.action_space.n
        env_params['max_action'] = 1
    elif not discrete_env: # Continuous
        env_params['actions'] = env.action_space.shape[0]
        env_params['max_action'] = env.action_space.high[0]

    # Observation space
    goal_env = isinstance(env.observation_space, gym.spaces.Dict)
    env_params['goal_env'] = goal_env
    if goal_env:
        env_params['observations'] = obs.shape[0]
    else:
        env_params['observations'] = obs.shape[0]

    return env_params
