import random
from torch.utils.data.sampler import Sampler


class OddEvenSampler(Sampler):
    def __init__(self, dataset):
        self.size = int(len(dataset))

    def __iter__(self):
        pattern = list(range(self.size))
        for i in range(len(pattern)):
            if i==len(pattern) - 1 and len(pattern)%2 != 0:
                break
            if pattern[i] % 2 == 0:
                pattern[i] += 1
            else:
                pattern[i] -= 1
        # modified pattern
        return iter(pattern)

    def __len__(self):
        return self.size
