import datetime
import time

import gym
import neptune
import numpy as np
import torch
from modules.actor import Actor
from modules.discriminator import Discriminator
from modules.gail import GAIL
from modules.helper import evaluate
from modules.trajectories import Expert
from sacred import Experiment

neptune.init(project_qualified_name='alvaro/circuit-irl',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzg2YzgxMDQtNjMwOS00NjNlLTgzZTAtYmRiYmVlYWNiNWUzIn0=',
             )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ex = Experiment()


@ex.config
def cfg():
    n_epochs = 100000
    eval_frequency = 100
    net_dim = (400, 300)
    lr = 0.0002
    betas = (0.5, 0.999)
    n_iter = 100
    batch_size = 512
    max_timesteps = 1000
    n_eval_episodes = 30
    expert = "TD3/trajectories/HumanoidPyBulletEnv-v0threshold1550traj50.pkl"
    env_name = 'HumanoidPyBulletEnv-v0'
    # env_name = "HalfCheetahPyBulletEnv-v0"
    # env_name = "SparseHalfCheetah-v0"


@ex.automain
def main(n_epochs, eval_frequency, net_dim, lr, betas, n_iter, batch_size, max_timesteps, n_eval_episodes, env_name,
         expert):
    import pybulletgym

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    seed = np.random.randint(0, 100000)
    torch.manual_seed(seed)
    np.random.seed(seed)

    neptune.create_experiment(
        params={'net_dim[0]': net_dim[0], 'net_dim[1]': net_dim[1], 'lr': lr, 'max_steps': max_timesteps,
                'betas[0]': betas[0], 'betas[1]': betas[1],
                "seed": seed}, tags=[env_name])

    actor = Actor(state_dim, action_dim, net_dim, lr, betas).to(device)
    expert = Expert(expert, state_dim)
    discriminator = Discriminator(state_dim, action_dim, net_dim, lr, betas, expert, actor, batch_size).to(device)
    gail = GAIL(actor, discriminator)

    # training procedure
    start_time = time.time()
    max_reward = float('-inf')

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
            neptune.log_metric('avg_rewards', x=epoch, y=avg_reward)
