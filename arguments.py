import argparse
import gym

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--algo', type=str, default='vpg',
                        help='which RL algorithm to use (default: vpg)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--image', action='store_true',
                        help='defines whether to solve the environment using pixels')
    parser.add_argument('--cuda', action='store_true',
                        help='defines whether to use GPU')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--vf-lr', type=float, default=1e-3)
    parser.add_argument('--train-v-iters', type=int, default=80)
    parser.add_argument('--max-ep-len', type=int, default=1000)
    # Q-learning
    parser.add_argument('--replay-size', type=int, default=int(1e6))
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--start-steps', type=int, default=10000)
    parser.add_argument('--q-lr', type=float, default=1e-3)
    parser.add_argument('--act-noise', type=float, default=0.1)
    # Variable
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--save-freq', type=int, default=None)
    parser.add_argument('--pi-lr', type=float, default=None)
    parser.add_argument('--train-pi-iters', type=int, default=None)
    # TRPO
    parser.add_argument('--delta', type=float, default=0.01)
    parser.add_argument('--damping-coeff', type=float, default=0.1)
    parser.add_argument('--cg-iters', type=int, default=10)
    parser.add_argument('--backtrack-iters', type=int, default=10)
    parser.add_argument('--backtrack-coeff', type=float, default=0.1)
    # PPO
    parser.add_argument('--clip-ratio', type=float, default=0.2)
    parser.add_argument('--target-kl', type=float, default=0.01)
    # TD3
    parser.add_argument('--target-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--policy-delay', type=int, default=2)
    # SAC
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.2)

    args = parser.parse_args()
    args = set_default_hyperparameters(args)
    return args

def set_default_hyperparameters(args):
    policy_gradient = ['reinforce', 'vpg', 'npg', 'trpo', 'ppo']
    q_learning = ['ddpg', 'td3', 'sac']

    if args.algo in policy_gradient:
        if args.steps is None:
            args.steps=4000
        if args.epochs is None:
            args.epochs=50
        if args.save_freq is None:
            args.save_freq=10
    elif args.algo in q_learning:
        if args.steps is None:
            args.steps=5000
        if args.epochs is None:
            args.epochs=100
        if args.save_freq is None:
            args.save_freq=1

    if args.algo == 'reinforce':
        if args.pi_lr is None:
            args.pi_lr = 3e-4
        if args.train_pi_iters is None:
            args.train_pi_iters = 1
    elif args.algo == 'vpg':
        if args.pi_lr is None:
            args.pi_lr = 3e-4
        if args.train_pi_iters is None:
            args.train_pi_iters = 1
    elif args.algo == 'npg':
        pass
    elif args.algo == 'trpo':
        pass
    elif args.algo == 'ppo':
        if args.pi_lr is None:
            args.pi_lr = 3e-4
        if args.train_pi_iters is None:
            args.train_pi_iters = 80
    elif args.algo == 'ddpg':
        if args.pi_lr is None:
            args.pi_lr = 1e-3
    elif args.algo == 'td3':
        if args.pi_lr is None:
            args.pi_lr = 1e-3
    elif args.algo == 'sac':
        pass
    return args
