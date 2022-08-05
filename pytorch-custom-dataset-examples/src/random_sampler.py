import random
from torch.utils.data.sampler import Sampler


class RandomSampler(Sampler):
    # Todo ensure that the variable is being updated
    shuffler_order_tracker = []

    @classmethod
    def update_tracker(cls, val):
        cls.shuffler_order_tracker += val

    @classmethod
    def get_tracker(cls):
        return cls.shuffler_order_tracker

    def __init__(self, dataset):
        # connect to add to metadata to set size (hardcoded for now)
        self.size = 50000

    def __iter__(self):
        self.pattern = list(range(self.size))
        random.shuffle(self.pattern)
        self.update_tracker(self.pattern)
        return iter(self.pattern)

    def __len__(self):
        return 50000

