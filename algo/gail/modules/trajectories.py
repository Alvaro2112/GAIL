import pickle as pk

import numpy as np
import torch


class Expert:
    def __init__(self, traj, state_dim):
        self.traj = pk.load(open(traj, 'rb'))
        self.size = len(self.traj)
        self.state_dim = state_dim

    def sample_state(self, batch_size):
        indexes = np.random.randint(0, self.size, size=batch_size)
        state = [None] * batch_size
        for index, i in enumerate(indexes):
            state[index] = self.traj[i][:self.state_dim]

        return torch.stack(state)

    def sample(self, batch_size):
        indexes = np.random.randint(0, self.size, size=batch_size)

        return self.traj[indexes]
