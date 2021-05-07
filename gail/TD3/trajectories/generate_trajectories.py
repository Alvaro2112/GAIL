import pickle

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
    n_trajectories = 1
    reward_threshold = 2300
    env_name = 'AntPyBulletEnv-v0'


@ex.automain
def main(n_trajectories, reward_threshold, env_name):
    env = gym.make(env_name)

    state_dimension = env.observation_space.shape[0]
    action_dimension = env.action_space.shape[0]
    actor = Actor(state_dimension, action_dimension)
    actor.load_state_dict(
        torch.load(
            "C://Users/alvar/Desktop/circuit_irl/circuit_irl/control_experiments/gail/TD3/_actorAntPyBulletEnv-v0",
            map_location=torch.device('cpu')))

    i = 0
    traj_pool = []

    while i < n_trajectories:
        tot_reward = 0
        state, done = env.reset(), False
        episode_traj = []
        seed = np.random.randint(0, 1000000)

        env.seed(seed)
        torch.manual_seed(seed)

        while not done:
            action = actor(torch.FloatTensor(state)).cpu().data.numpy()
            pair = torch.cat([torch.Tensor(state), torch.Tensor(action)], 0)
            episode_traj.append(pair)

            # act
            state, reward, done, _ = env.step(action)

            tot_reward += reward

        print("seed:" + str(seed) + "     avg_reward:" + str(tot_reward))
        if tot_reward >= reward_threshold:
            i += 1
            traj_pool.extend(episode_traj)

    traj_pool = torch.stack(traj_pool)

    file = open(env_name + "thresholdx" + str(reward_threshold) + "traj" + str(n_trajectories) + ".pkl", 'wb')
    pickle.dump(traj_pool, file)
