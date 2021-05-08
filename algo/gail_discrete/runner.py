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


def objective(trial):
    n_epochs = 20000
    eval_frequency = 100
    max_steps = 300
    actor_net_dim = (128, 128)
    critic_net_dim = (128, 128)
    dsicriminator_net_dim = (128, 128)
    lr = trial.suggest_float("lr", 0, 0.01)
    gamma = trial.suggest_float("gamma", 0.9, 1)
    tau = trial.suggest_float("tau", 0.9, 1)
    grad_clip = 40
    batch_size = 2 ** trial.suggest_int('batch_size', 4, 8)
    betas = (0.9, 0.999)
    entropy_weight = trial.suggest_float("entropy_weight", 0, 0.1)
    min_buffer_size = 2048
    clip = trial.suggest_float("clip", 0, 0.5)
    ppo_updates = trial.suggest_int('ppo_updates', 1, 20)
    expert = trial.suggest_categorical("expert", ["1", "3", "10"])
    value_coef = trial.suggest_float("value_coef", 0, 1)
    activation = "tanh"
    env_name = "LunarLander-v2"
    record = True

    if expert == "1":
        expert = 1
    elif expert == "3":
        expert = 3
    else:
        expert = 10

    seed = 99
    discriminator_updates = 1

    expert, activation = initiate_run(env_name, actor_net_dim, critic_net_dim, dsicriminator_net_dim,
                                      lr, gamma, tau, grad_clip, batch_size, entropy_weight,
                                      min_buffer_size, clip, ppo_updates, discriminator_updates, expert, activation,
                                      value_coef, betas, max_steps, seed, "", record)

    env = Env(env_name)
    actor = Actor(env, actor_net_dim, activation)
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
    sys.stdout.flush()
    return epoch_to_best


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
    env_name = "LunarLander-v2"
    # env_name = "CartPole-v0"
    # env_name = "Acrobot-v1"
    # env_name = "SparseMountainCar-v0"
    # env_name = "MountainCar-v0"
    tag = ""
    record = True


@ex.automain
def main(env_name, n_epochs, eval_frequency, actor_net_dim, critic_net_dim, dsicriminator_net_dim,
         lr, gamma, tau, grad_clip, batch_size, entropy_weight, min_buffer_size, clip, ppo_updates,
         expert, activation, value_coef, betas, max_steps, tag, record):

    seed = np.random.randint(0, 1000000)
    seed = 99

    discriminator_updates = 1

    expert, activation = initiate_run(env_name, actor_net_dim, critic_net_dim, dsicriminator_net_dim,
                                      lr, gamma, tau, grad_clip, batch_size, entropy_weight,
                                      min_buffer_size, clip, ppo_updates, discriminator_updates, expert, activation,
                                      value_coef, betas, max_steps, seed, tag, record)

    env = Env(env_name)
    actor = Actor(env, actor_net_dim, activation)
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

    #study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(
    #n_startup_trials=5, n_warmup_steps=5000))  # Create a new study.
    #study.optimize(objective, n_trials=1000)
