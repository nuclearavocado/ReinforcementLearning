# Local
import numpy as np
from utils import printISO8601

class BaseLogger:
    """
        Saves variables into a dictionary at various resolutions, as:
         - episode averages
         - epoch averages
        It then prints out this information every epoch.
    """
    def __init__(self, reward_threshold=None):
        self.reward_threshold = reward_threshold
        self.vars, self.episode_total, self.epoch_average = {}, {}, {}
        self.epoch, self.episode, self.iteration = 0, 0, 0
        self.episode_prev, self.epoch_prev = 0, 0
        self.episode_start_idx = 0
        self.episode_ptr = 0 # No. episodes in previous epoch

    def add_scalars(self, keys, values):
        # If new episode, get variable averages
        if self.episode > self.episode_prev:
            self.finish_episode()

        '''
            Add scalar(s) to buffer
        '''
        # This function allows multiple scalar keys and values to be passed in
        # as a list. If only a single key and value is passed in, convert to a
        # list, then loop through the key-value pairs as normal.
        if type(values) != list:
            keys = [keys]
            values = [values]
        for key, value in zip(keys, values):
            # Create the scalar if it doesn't already exist
            if key not in self.vars:
                self.vars[key] = []
            # Add the scalar to the list of tracked variables
            self.vars[key].append(value)

    def finish_epoch(self):
        # Finish up the episode first
        self.finish_episode()
        # Calculate variable averages for the epoch
        for key, value in self.episode_total.items():
            self.epoch_average[key] = sum(value)/len(value)
        self.dump_logs() # dump logs
        self.vars, self.episode_total = {}, {} # reset buffers
        self.episode_start_idx = 0 # reset pointer
        self.epoch_prev = self.epoch # update epoch number

    def finish_episode(self):
        # Calculate variable averages for the episode
        for key, value in self.vars.items():
            # Create the averages if they don't already exist
            if key not in self.episode_total:
                self.episode_total[key] = []
            # The entire history of the variable over the course of the epoch
            # is stored in the buffer, but we only want the values for this
            # episode.
            v = value[self.episode_start_idx:]
            # Calculate the total episode reward and add to a list of total
            # episode rewards over the course of the epoch
            self.episode_total[key].append(sum(v))
        self.episode_start_idx += len(v) # update the episode index pointer
        self.episode_prev = self.episode # update episode number

    def dump_logs(self):
        raise NotImplementedError

class TabularLogger(BaseLogger):
    def __init__(self, reward_threshold=None):
        super().__init__(reward_threshold=reward_threshold)

    def dump_logs(self):
        # Print epoch average reward and if environment is completed to console
        print(f"reward {self.epoch_average['reward']:.2f}")
        # Show the environment is solved if this criterion exists
        if (self.reward_threshold is not None) and (self.epoch_average['reward'] >= self.reward_threshold):
            print()
            print("Solved!")

class TensorBoardLogger(TabularLogger):
    def __init__(self, args):
        from torch.utils.tensorboard import SummaryWriter
        super().__init__(reward_threshold=args.reward_threshold)
        dir_name = f"./graphs/{printISO8601().replace(':', '_')}_{args.algo}_{args.env}"
        self.writer = SummaryWriter(log_dir=dir_name)

    def dump_logs(self):
        # Print to console
        super().dump_logs()
        # Add to TensorBoard
        for k, value in self.episode_total.items():
            k = 'episode total ' + k
            for i, v in enumerate(value):
                self.writer.add_scalar(k, v, self.episode_ptr + i)
        self.episode_ptr = self.episode
        for k, v in self.epoch_average.items():
            k = 'epoch average ' + k
            self.writer.add_scalar(k, v, self.epoch)
