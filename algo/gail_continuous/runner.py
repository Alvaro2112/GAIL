import itertools
import sys

import neptune
import numpy as np
import torch
from modules.Env import Env
from modules.actor import Actor
from modules.critic import Critic
from modules.discriminator import Discriminator
from modules.gail import GAIL
from modules.helper import initiate_run
from modules.ppo_agent import Agent
from modules.replay_buffer import PPOMemory
from sacred import Experiment

neptune.init(project_qualified_name='alvaro/circuit-irl',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzg2YzgxMDQtNjMwOS00NjNlLTgzZTAtYmRiYmVlYWNiNWUzIn0=',
             )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ex = Experiment()

@ex.config
def cfg():
    n_epochs = 20000
    eval_frequency = 100
    max_steps = 300
    actor_net_dim = (256, 256)
    critic_net_dim = (256, 256)
    dsicriminator_net_dim = (256, 256)
    lr = 1.250237560842974E-4
    gamma = 0.9083367314790233
    tau = 0.9070183761981814
    grad_clip = 40
    batch_size = 32
    betas = (0.9, 0.999)
    entropy_weight = 4.034966814115934E-4
    min_buffer_size = 2048
    clip = 0.281987817625237
    ppo_updates = 4
    expert = 10
    value_coef = 0.8431282745909283
    activation = "tanh"
    env_name = "HalfCheetahPyBulletEnv-v0"
    tag = ""
    record = False


@ex.automain
def main(env_name, n_epochs, eval_frequency, actor_net_dim, critic_net_dim, dsicriminator_net_dim,
         lr, gamma, tau, grad_clip, batch_size, entropy_weight, min_buffer_size, clip, ppo_updates,
         expert, activation, value_coef, betas, max_steps, tag, record):

    seed = np.random.randint(0, 1000000)
    import pybulletgym
    discriminator_updates = 1

    expert, activation = initiate_run(env_name, actor_net_dim, critic_net_dim, dsicriminator_net_dim,
                                      lr, gamma, tau, grad_clip, batch_size, entropy_weight,
                                      min_buffer_size, clip, ppo_updates, discriminator_updates, expert, activation,
                                      value_coef, betas, max_steps, seed, tag, record)

    env = Env(env_name)
    actor = Actor(env, actor_net_dim, activation, env.env.action_space.high, env.env.action_space.low)
    critic = Critic(env, critic_net_dim, activation)
    discriminator = Discriminator(env, dsicriminator_net_dim, lr, batch_size, activation, betas)
    agent = Agent(gamma, clip, actor, critic, lr, batch_size, grad_clip, entropy_weight, value_coef, betas)
    memory = PPOMemory(gamma, tau)

    args = [min_buffer_size, eval_frequency,
            ppo_updates, discriminator_updates, expert, seed]

    gail = GAIL(env, actor, critic, discriminator, agent, memory, *args)
    epoch_to_best = gail.update(n_epochs, max_steps, record)
    if record:
        neptune.log_metric('best_epoch', epoch_to_best)
        neptune.stop()
