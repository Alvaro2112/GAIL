import datetime
import time

import gym
import numpy as np
import torch
from dd.GAIL.algo.gail.helper.config import *
from dd.GAIL.helper.common import *
from modules.actor import Actor
from modules.discriminator import Discriminator
from modules.gail import GAIL
from modules.helper import evaluate
from modules.trajectories import Expert

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    args = setup_parser()
    config = load_config_from_yaml(args.config)
    config.set_args(args)

    env_name = config['env'].get()
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    seed = np.random.randint(0, 100000)
    torch.manual_seed(seed)
    np.random.seed(seed)

    actor = Actor(state_dim, action_dim, config['actor_net_dim'].get(), config['lr'].get(), config['betas'].get()).to(
        device)
    expert = Expert(config['expert'].get(), state_dim)
    discriminator = Discriminator(state_dim, action_dim, config['discriminator_net_dim'].get(), config['lr'].get(),
                                  config['betas'].get(),
                                  expert, actor, config['batch_size'].get()).to(device)
    gail = GAIL(actor, discriminator)

    # training procedure
    start_time = time.time()
    max_reward = float('-inf')

    n_epochs = config['n_epochs'].get()
    n_iter = config['n_iter'].get()
    eval_frequency = config['eval_frequency'].get()
    n_eval_episodes = config['n_eval_episodes'].get()
    max_timesteps = config['max_timesteps'].get()

    for epoch in range(1, n_epochs + 1):

        # update policy n_iter times
        params = gail.update(n_iter)

        if epoch % eval_frequency == 0:

            # evaluate in environment
            avg_reward = evaluate(n_eval_episodes, env, max_timesteps, gail)

            if avg_reward > max_reward:
                torch.save(params, "./actor_weights" + str(env_name) + str(seed))

            max_reward = max(avg_reward, max_reward)

            print("Epoch: {}\tAvg Reward: {} in {}".format(epoch, avg_reward, datetime.timedelta(
                seconds=(time.time() - start_time).__round__(0))))


if __name__ == '__main__':
    main()
