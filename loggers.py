# Local
from utils import printISO8601

class BaseLogger:
    def __init__(self):
        self.vars = {}
        self.buffer = {}

    def add_scalar(self, key, value):
        self.vars[key] = value

    def average_scalar(self, key, buffer_length=10):
        value = self.vars[key]
        average_key = 'average' + ' ' + key

        # Create an empty list if the key is not being tracked
        if not average_key in self.buffer:
            self.buffer[average_key] = []

        # FIFO buffer of size `buffer_length`
        if len(self.buffer[average_key]) < buffer_length:
            self.buffer[average_key].append(value)
        else:
            self.buffer[average_key].pop(0) # remove first element
            self.buffer[average_key].append(value) # add the latest value
        self.vars[average_key] = sum(self.buffer[average_key])/len(self.buffer[average_key])

    def dump_logs(self, iteration, log_interval):
        raise NotImplementedError

class TabularLogger(BaseLogger):
    def __init__(self):
        super().__init__()

    def dump_logs(self, iteration, log_interval, reward_threshold=None):
        if iteration % log_interval == 0:
            print(f'iteration: {iteration}\t', end='')
            for key, value in self.vars.items():
                if isinstance(value, float):
                    print(f'{key}: {value:.2f}\t', end='')
                else:
                    print(f'{key}: {value}\t', end='')
            print()
            if (reward_threshold is not None) and (self.vars['episode reward'] >= reward_threshold):
                print("Solved!")

class TensorBoardLogger(TabularLogger):
    def __init__(self, args):
        from torch.utils.tensorboard import SummaryWriter
        super().__init__()
        dir_name = f"./graphs/{printISO8601().replace(':', '_')}_{args.algo}_{args.env_name}_batch_size_{args.n_batches}_{args.buffer}_{args.model}_{args.policy}"
        self.writer = SummaryWriter(log_dir=dir_name)

    def dump_logs(self, iteration, log_interval, reward_threshold=None):
        for key, value in self.vars.items():
            self.writer.add_scalar(key, value, iteration)
        super().dump_logs(iteration, log_interval, reward_threshold=reward_threshold)
