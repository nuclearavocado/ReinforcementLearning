import argparse
import gym

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str,
                                 default="CartPole-v1",
                                 help="(str): Gym environment to train the" +
                                      "agent on. (default: CartPole-v1)")
    parser.add_argument("--algo", type=str,
                                  default="reinforce",
                                  help="(str): Which reinforcement learning " +
                                       "algorithm to use. (default: reinforce)")
    parser.add_argument("--render", action="store_true",
                                    help="(bool): Whether to render the " +
                                         "environment (default: False)")
    parser.add_argument("--render-interval", type=int,
                                             default=1,
                                             help="(int): How often to " +
                                             "render the environment. " +
                                             "(default: 1)")
    parser.add_argument("--image", action="store_true",
                                   help="(bool): Defines whether to solve " +
                                        "the environment by observing " +
                                        "pixels instead of the raw state. " +
                                        "(default: False)")
    parser.add_argument("--cuda", action="store_true",
                                  help="(bool): Uses the GPU if true. " +
                                       "(default: False)")
    parser.add_argument("--hid", type=int,
                                 default=64,
                                 help="(int): Number of neurons in the " +
                                      "hidden layers of the MLP. (default: 64)")
    parser.add_argument("--l", type=int,
                               default=2,
                               help="(int): Number of hidden layers in the " +
                                    "MLP. (default: 2)")
    parser.add_argument("--gamma", type=float,
                                   default=0.99,
                                   help="(float): Discount factor (always " +
                                        "between 0 and 1). (default: 0.99)")
    parser.add_argument("--lam", type=float,
                                 default=0.97,
                                 help="(float): Lambda for GAE-Lambda. " +
                                      "(Always between 0 and 1, close to " +
                                      "1.) (default: 0.97)")
    parser.add_argument("--seed", "-s", type=int,
                                        default=69,
                                        help="(int): Seed for random number " +
                                             "generators. (default: 69)")
    parser.add_argument("--v-lr", type=float,
                                  default=1e-3,
                                  help="(float): Learning rate for value " +
                                        "function optimizer. (default: 10^-3)")
    parser.add_argument("--train-v-iters", type=int,
                                           default=80,
                                           help="(int): Number of gradient " +
                                                "descent steps to take on " +
                                                "value function per epoch. " +
                                                "(default: 80)")
    parser.add_argument("--max-ep-len", type=int,
                                        default=1000,
                                        help="(int): Maximum length of " +
                                             "trajectory/episode/rollout. " +
                                             "(default: 1000)")
    parser.add_argument("--buffer", type=str,
                                    default="default",
                                    help="(str): Type of buffer to use. " +
                                         "(default: 'default')")
    # Q-learning
    parser.add_argument("--replay-size", type=int,
                                         default=int(1e6),
                                         help="(int): Maximum length of " +
                                              "replay buffer. (default: 10^6)")
    parser.add_argument("--polyak", type=float,
                                    default=0.995,
                                    help="(float): Interpolation factor in " +
                                         "polyak averaging for target " +
                                         "networks. (Always between 0 and 1, " +
                                         "usually close to 1.) " +
                                         "(default: 0.995)")
    parser.add_argument("--batch-size", type=int,
                                        default=100,
                                        help="(int): Minibatch size for SGD. " +
                                             "(default: 100)")
    parser.add_argument("--start-steps", type=int,
                                         default=10000,
                                         help="(int): Number of steps for " +
                                              "uniform-random action " +
                                              "selection, before running " +
                                              "real policy. Helps " +
                                              "exploration. (default: 10000)")
    parser.add_argument("--q-lr", type=float,
                                  default=1e-3,
                                  help="(float): Learning rate for " +
                                       "Q-networks. (default: 10^-3)")
    parser.add_argument("--act-noise", type=float,
                                       default=0.1,
                                       help="(float): Stddev for Gaussian " +
                                            "exploration noise added to " +
                                            "policy at training time. (At " +
                                            "test time, no noise is added.) " +
                                            "(default: 0.1)")
    parser.add_argument("--update-after", type=int,
                                          default=1000,
                                          help="(int): Number of env " +
                                               "interactions to collect " +
                                               "before starting to do " +
                                               "gradient descent updates. " +
                                               "Ensures replay buffer is " +
                                               "full enough for useful " +
                                               "updates. (default: 1000)")
    parser.add_argument("--train-q-iters", type=int,
                                          default=50,
                                          help="(int): Number of env " +
                                               "interactions that should " +
                                               "elapse between gradient " +
                                               "descent updates. " +
                                               "Note: Regardless of how long " +
                                               "you wait between updates, " +
                                               "the ratio of env steps to " +
                                               "gradient steps is currently " +
                                               "locked to 1. (default: 50)")
    parser.add_argument("--num-test-episodes", type=int,
                                               default=10,
                                               help="(int): Number of " +
                                                    "episodes to test the " +
                                                    "deterministic policy at " +
                                                    "the end of each epoch. " +
                                                    "(default: 10)")
    # Variable
    parser.add_argument("--steps", type=int,
                                   default=None,
                                   help="(int): Number of steps of " +
                                        "interaction (state-action pairs) " +
                                        "for the agent to take in the " +
                                        "environment for each epoch. " +
                                        "(default: varies by algorithm)")
    parser.add_argument("--epochs", type=int,
                                    default=None,
                                    help="(int): Number of epochs to run and " +
                                         "train agent. " +
                                         "(default: varies by algorithm)")
    parser.add_argument("--save-freq", type=int,
                                       default=None,
                                       help="(int): How often (in terms of " +
                                            "gap between epochs) to save the " +
                                            "current policy and value " +
                                            "function. " +
                                            "(default: varies by algorithm)")
    parser.add_argument("--pi-lr", type=float,
                                   default=None,
                                   help="(float): Learning rate for policy " +
                                        "optimizer. " +
                                        "(default: varies by algorithm)")
    parser.add_argument("--train-pi-iters", type=int,
                                            default=None,
                                            help="(int): Maximum number of " +
                                                 "gradient descent steps to " +
                                                 "take on policy loss per " +
                                                 "epoch. (Early stopping may " +
                                                 "cause optimizer to take " +
                                                 "fewer than this.) " +
                                                 "(default: varies by " +
                                                 "algorithm)")
    # TRPO
    parser.add_argument("--delta", type=float,
                                   default=0.01,
                                   help="(float): . (default: 0.01)")
    parser.add_argument("--damping-coeff", type=float,
                                           default=0.1,
                                           help="(float): . (default: 0.1)")
    parser.add_argument("--cg-iters", type=int,
                                      default=10,
                                      help="(int): . (default: 10)")
    parser.add_argument("--backtrack-iters", type=int,
                                             default=10,
                                             help="(int): . (default: 10)")
    parser.add_argument("--backtrack-coeff", type=float,
                                             default=0.1,
                                             help="(float): . (default: 0.1)")
    # PPO
    parser.add_argument("--clip-ratio", type=float,
                                        default=0.2,
                                        help="(float): Hyperparameter for " +
                                             "clipping in the policy " +
                                             "objective. Roughly: how far " +
                                             "can the new policy go from the " +
                                             "old policy while still " +
                                             "profiting (improving the " +
                                             "objective function)? The new " +
                                             "policy can still go farther " +
                                             "than the clip_ratio says, but " +
                                             "it doesn't help on the " +
                                             "objective anymore. (Usually " +
                                             "small, 0.1 to 0.3.) Typically " +
                                             "denoted by epsilon. " +
                                             "(default: 0.2)")
    parser.add_argument("--target-kl", type=float,
                                       default=0.01,
                                       help="(float): Roughly what KL " +
                                            "divergence we think is " +
                                            "appropriate between new and old " +
                                            "policies after an update. This " +
                                            "will get used for early " +
                                            "stopping. (Usually small, 0.01 " +
                                            "or 0.05.). (default: 0.01)")
    # TD3
    parser.add_argument("--target-noise", type=float,
                                          default=0.2,
                                          help="(float): Stddev for " +
                                               "smoothing noise added to " +
                                               "target policy. (default: 0.2)")
    parser.add_argument("--noise-clip", type=float,
                                        default=0.5,
                                        help="(float): Limit for absolute " +
                                             "value of target policy " +
                                             "smoothing noise. (default: 0.5)")
    parser.add_argument("--policy-delay", type=int,
                                          default=2,
                                          help="(int): Policy will only be " +
                                               "updated once every " +
                                               "policy_delay times for each " +
                                               "update of the Q-networks. " +
                                               "(default: 2)")
    # SAC
    parser.add_argument("--alpha", type=float,
                                   default=0.2,
                                   help="(float): Entropy regularization " +
                                        "coefficient. (Equivalent to inverse " +
                                        "of reward scale in the original SAC " +
                                        "paper.) (default: 0.2)")
    # HER
    parser.add_argument("--replay-k", type=int,
                                      default=4,
                                      help="(int): the ratio between HER " +
                                           "replays and regular replays " +
                                           "(e.g. k = 4 => 4 times as many " +
                                           "HER replays as regular replays " +
                                           "are used). (default: 4)")

    args = parser.parse_args()
    args = set_default_hyperparameters(args)
    return args

def set_default_hyperparameters(args):
    # Convert to lowercase
    args.algo = args.algo.lower()
    args.buffer = args.buffer.lower()
    args.policy_gradient = ["reinforce", "vpg", "npg", "trpo", "ppo"]
    args.q_learning = ["dqn", "ddpg", "td3", "sac"]

    if args.algo in args.policy_gradient:
        if args.steps is None:
            args.steps=4000
        if args.epochs is None:
            args.epochs=50
        if args.pi_lr is None:
            args.pi_lr = 3e-4
        if args.save_freq is None:
            args.save_freq=10
    elif args.algo in args.q_learning:
        if args.steps is None:
            args.steps=5000
        if args.epochs is None:
            args.epochs=100
        if args.pi_lr is None:
            args.pi_lr = 1e-3
        if args.save_freq is None:
            args.save_freq=1

    if args.algo in ["reinforce", "vpg"]:
        if args.train_pi_iters is None:
            args.train_pi_iters = 1
    elif args.algo == "npg":
        pass
    elif args.algo == "trpo":
        pass
    elif args.algo == "ppo":
        if args.train_pi_iters is None:
            args.train_pi_iters = 80
    return args
