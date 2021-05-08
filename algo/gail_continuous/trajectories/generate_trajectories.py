import gym
import numpy as np
import torch
import torch.nn.functional as F
from sacred import Experiment
from torch import nn

ex = Experiment()


class Actor(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


@ex.config
def cfg():
    n_trajectories = 10
    reward_threshold = 1750


@ex.automain
def main(n_trajectories, reward_threshold):
    env = gym.make('HalfCheetahPyBulletEnv-v0')

    state_dimension = env.observation_space.shape[0]
    action_dimension = env.action_space.shape[0]
    actor = Actor(state_dimension, action_dimension)
    actor.load_state_dict(
        torch.load("C://Users/alvar/Desktop/circuit_irl/control_experiments/gail/TD3/saved_model/_actor",
                   map_location=torch.device('cpu')))

    i = 0
    step = 0
    action_trajectories = np.zeros((1000 * n_trajectories, action_dimension))
    state_trajectories = np.zeros((1000 * n_trajectories, state_dimension))

    while i < n_trajectories:
        tot_reward = 0
        state, done = env.reset(), False

        while not done:
            action = actor(torch.FloatTensor(state)).cpu().data.numpy()  # add noise still?

            action_trajectories[step] = action
            state_trajectories[step] = state

            # act
            state, reward, done, _ = env.step(action)

            tot_reward += reward
            step += 1
        if tot_reward >= reward_threshold:
            i += 1
        else:
            step -= 1000
    np.save("action_trajectoriess.csv", action_trajectories)
    np.save("state_trajectoriess.csv", state_trajectories)

    # TODO add evaluation method
