import torch
from torch.utils.data import Sampler


class PulseSampler(Sampler):
    """
    Sampler to pulse dataset.
    """

    def __init__(self, end_idx, seq_len, random=False):
        """
        samples indexes of video sequence
        :param end_idx: list of end indexes of every sequence, 0 at first position
        :param seq_len: length of output sequence (window size)
        """

        indices = []
        for i in range(len(end_idx) - 1):
            start = end_idx[i]
            end = end_idx[i]+end_idx[i + 1] - seq_len
            # print(start, end, end_idx)
            indices.append(torch.arange(start, end, seq_len))  # indexes uniformly distributed between start and end  -- Seq_len
            # print(indices)

            # non overlaping window
        indices = torch.cat(indices)
        # print(indices)
        self.indices = indices
        self.random = random

    def __len__(self):

        return len(self.indices)

    def __iter__(self):
        if self.random:
            indices = self.indices[torch.randperm(len(self.indices))]  # random permutation of available indexes
            return iter(indices.tolist())
        else:
            indices = self.indices  # available indexes without shuffling
            print('hello', indices)
            return iter(indices.tolist())


