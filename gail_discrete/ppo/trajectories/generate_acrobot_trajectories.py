import pickle

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from sacred import Experiment
from torch.distributions import Categorical

ex = Experiment()


class Policy(nn.Module):
    def __init__(self, s_size=6, h_size=32, a_size=3):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item() - 1, m.log_prob(action)


torch.manual_seed(0)  # set random seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_dict = torch.load('C://Users/alvar/Desktop/circuit_irl/circuit_irl/control_experiments/gail_discrete/ppo'
                        '/saved_model/acrobot.pth')

policy = Policy()
policy.load_state_dict(state_dict)
policy = policy.to(device)


@ex.config
def cfg():
    n_trajectories = 100
    reward_threshold = -100


@ex.automain
def main(n_trajectories, reward_threshold):
    env = gym.make('Acrobot-v1')
    i = 0
    traj_pool = []

    while i < n_trajectories:

        tot_reward = 0
        state, done = env.reset(), False

        episode_traj = []

        while not done:
            action, _ = policy.act(state)
            episode_traj.append([state, action])

            state, reward, done, _ = env.step(action)
            tot_reward += reward
            if done:
                break
        if tot_reward >= reward_threshold:
            traj_pool.extend(episode_traj)
            i += 1
            print(n_trajectories - i)

    file = open('acrobot_100.pkl', 'wb')
    pickle.dump(traj_pool, file)
