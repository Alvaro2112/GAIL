import pickle

import gym
import torch
from sacred import Experiment
from torch import nn
from torch.distributions import Categorical

ex = Experiment()


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.action = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.value = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def build_distribution(self, state):
        # get probabilitys for each action
        probabilitys = self.action(state)

        # Build a multinomial distribution
        return Categorical(probabilitys)

    def act(self, state):
        state = torch.from_numpy(state).float()

        distribution = self.build_distribution(state)

        # Sample an action from distribution
        action = distribution.sample()

        # Returns value in tensor, if any
        return action.item()


@ex.config
def cfg():
    n_trajectories = 3
    reward_threshold = 200


@ex.automain
def main(n_trajectories, reward_threshold):
    env = gym.make("CartPole-v0")

    state_dimension = env.observation_space.shape[0]
    action_dimension = env.action_space.n

    actor = Actor(state_dimension, action_dimension)
    actor.load_state_dict(
        torch.load("C://Users/alvar/Desktop/circuit_irl/circuit_irl/control_experiments/gail_discrete/ppo/saved_model"
                   "/cart_pole_actor.pth"))

    i = 0

    traj_pool = []

    while i < n_trajectories:

        tot_reward = 0
        state, done = env.reset(), False

        step = 0
        episode_traj = []
        while not done:
            step += 1
            action = actor.act(state)
            episode_traj.append([state, action])
            state, reward, done, _ = env.step(action)
            tot_reward += reward

        if tot_reward >= reward_threshold:
            traj_pool.extend(episode_traj)
            i += 1
            print(n_trajectories - i)

    file = open('cart_pole_3.pkl', 'wb')
    pickle.dump(traj_pool, file)
