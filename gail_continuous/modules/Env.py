from abc import ABC

import gym


class Env(gym.Env, ABC):
    def __init__(self, env):
        self.env = gym.make(env)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

    def get_action_dim(self):
        return self.action_dim

    def get_state_dim(self):
        return self.state_dim

    def get_env(self):
        return self.env
