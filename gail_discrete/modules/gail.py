import pickle
import sys
import numpy as np
import torch
from modules.helper import get_reward, prepare_data, log_metrics
import optuna
import neptune

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GAIL:
    def __init__(self, env, actor, critic, discriminator, agent, memory, min_buffer_size, evaluation_interval,
                 ppo_updates, discriminator_updates, expert, seed, trial=None):

        self.action_dim = env.get_action_dim()
        self.state_dim = env.get_state_dim()
        self.env = env.get_env()
        self.env.seed(seed)
        self.trial = trial

        self.min_buffer_size = min_buffer_size
        self.ppo_updates = ppo_updates
        self.discriminator_updates = discriminator_updates

        self.actor = actor
        self.critic = critic
        self.discriminator = discriminator
        self.agent = agent
        self.memory = memory

        self.pool = pickle.load(open(expert, 'rb'))
        self.evaluation_interval = evaluation_interval

    def update(self, n_iter, max_steps, record):

        tot_reward = 0
        tot_policy_loss = 0
        tot_discr_loss = 0
        episodes_in_stats = 0
        epoch_to_best = -1
        rewards = []
        best_avg = -999999

        for epoch in range(1, n_iter + 1):

            obs = self.env.reset()
            episode_reward = 0
            ep_discr_loss = 0
            ep_policy_loss = 0
            for step in range(1, max_steps + 1):

                with torch.no_grad():
                    state = torch.FloatTensor(obs)
                    action, logpa_tensor, _ = self.actor.act(state)
                    value = self.critic(torch.FloatTensor(obs))
                    custom_reward = get_reward(obs, action, self.discriminator)

                next_obs, reward, done, _ = self.env.step(action)

                self.memory.store(state, action, custom_reward, value, logpa_tensor.item())

                episode_reward += reward
                obs = next_obs

                time_to_optimize = len(self.memory) == self.min_buffer_size
                timeout = step == max_steps

                if done or timeout or time_to_optimize:

                    if done:
                        value = 0
                    else:
                        statess = torch.FloatTensor(next_obs)
                        with torch.no_grad():
                            value = self.critic(statess).item()

                    self.memory.finish_path(value)

                    if time_to_optimize:
                        data = prepare_data(self.memory.get())
                        ep_discr_loss += self.discriminator.update(self.pool,
                                                                   data)
                        ep_policy_loss += self.agent.update(self.ppo_updates, data)

                if done:
                    tot_reward += episode_reward
                    tot_discr_loss += ep_discr_loss
                    tot_policy_loss += ep_policy_loss
                    rewards.append(episode_reward)
                    episodes_in_stats += 1
                    break

            # Log rewards and metrics
            if epoch % self.evaluation_interval == 0:
                log_metrics(self.evaluation_interval, tot_reward, epoch, tot_policy_loss, tot_discr_loss,
                            episodes_in_stats, record)
                tot_reward = 0
                tot_policy_loss = 0
                tot_discr_loss = 0
                episodes_in_stats = 0

            if len(rewards) >= 1000:
                avg_reward = np.array(rewards[epoch - 1000: epoch]).mean()
                if avg_reward > best_avg:
                    best_avg = avg_reward
                    if record:
                        neptune.log_metric('best_avg', y=best_avg, x=epoch)
                if epoch % self.evaluation_interval == 0 and not self.trial is None:
                    self.trial.report(best_avg, step=epoch)
                    if self.trial.should_prune():
                        neptune.stop()
                        raise optuna.TrialPruned()

        return best_avg
