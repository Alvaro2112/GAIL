import gym
import numpy as np
import torch
from modules.actor import Actor
from modules.helper import evaluate

env_name = "HalfCheetahPyBulletEnv-v0"
net_dim = (400, 300)
path = "./actor_weights97118"
for i in range(5):
    seed = np.random.randint(0, 1000000)
    env = gym.make(env_name)

    env.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim, net_dim)
    actor.load_state_dict(torch.load(path))
    avg_reward = evaluate(1, env, 1000, actor)

    print("seed:" + str(seed) + "     avg_reward:" + str(avg_reward))
