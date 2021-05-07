import copy

import gym
import neptune
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sacred import Experiment
from torch import nn

ex = Experiment()

neptune.init(project_qualified_name='alvaro/circuit-irl',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzg2YzgxMDQtNjMwOS00NjNlLTgzZTAtYmRiYmVlYWNiNWUzIn0=',
             )


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


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)

        c1 = F.relu(self.l1(x))
        c1 = F.relu(self.l2(c1))
        c1 = self.l3(c1)

        c2 = F.relu(self.l4(x))
        c2 = F.relu(self.l5(c2))
        c2 = self.l6(c2)

        return c1, c2

    def critic1(self, state, action):
        x = torch.cat([state, action], 1)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x


class replay_buffer:

    def __init__(self, state_dimension, action_dimension, max_size=int(1e6)):
        self.p_states = np.zeros((max_size, state_dimension))
        self.states = np.zeros((max_size, state_dimension))
        self.actions = np.zeros((max_size, action_dimension))
        self.rewards = np.zeros((max_size, 1))
        self.dones = np.zeros((max_size, 1))
        self.index = 0
        self.size = 0
        self.max_size = max_size

    def add(self, p_state, state, action, reward, done):
        self.p_states[self.index] = p_state
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = 1 - done
        self.index = (self.index + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indexes = np.random.randint(0, self.size, size=batch_size)

        return self.p_states[indexes], self.states[indexes], self.actions[indexes], self.rewards[indexes], self.dones[
            indexes]


def to_tensor(p_state, state, action, reward, done, device):
    p_state = torch.FloatTensor(p_state).to(device)
    state = torch.FloatTensor(state).to(device)
    reward = torch.FloatTensor(reward).to(device)
    action = torch.FloatTensor(action).to(device)
    done = torch.FloatTensor(done).to(device)

    return p_state, state, action, reward, done


@ex.config
def cfg():
    start_exploiting = 10000
    max_steps = 1000000
    plot_interval = 25
    noise = 0.2
    noise_clip = 0.5
    noise_std = 0.1
    update_interval = 2
    evaluate_iterations = 10
    memory_batch_size = 100
    discount_factor = 0.99
    tau = 0.005
    lr = 1e-3
    env_name = 'HalfCheetahPyBulletEnv-v0'


@ex.automain
def main(start_exploiting, max_steps, noise, noise_clip, noise_std, update_interval, memory_batch_size, discount_factor,
         tau, lr, plot_interval, env_name, evaluate_iterations):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = np.random.randint(0, 100000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    max_avg = -9999

    neptune.create_experiment(
        params={'lr': lr, 'max_steps': max_steps, 'discount_factor': discount_factor, 'tau': tau,
                'noise': noise, 'noise_std': noise_clip, 'noise_clip': noise_clip,
                'memory_batch_size': memory_batch_size,
                "seed": seed}, tags=[env_name])

    env = gym.make(env_name)

    state_dimension = env.observation_space.shape[0]
    action_dimension = env.action_space.shape[0]

    actor = Actor(state_dimension, action_dimension).to(device)
    actor_target = copy.deepcopy(actor).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)

    critic = Critic(state_dimension, action_dimension).to(device)
    critic_target = copy.deepcopy(critic).to(device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    memory = replay_buffer(state_dimension, action_dimension)

    state = env.reset()

    tot_reward = 0
    step = 0
    episode_number = 0
    steps_in_episode = 0
    reward_avg = 0

    for j in range(max_steps):

        step += 1
        steps_in_episode += 1

        if j < start_exploiting:
            action = env.action_space.sample()
        else:
            statex = torch.FloatTensor(state).to(device)
            action = actor(statex).cpu().data.numpy()
            action = action + np.random.normal(0, env.action_space.high[0] * noise_std, size=action_dimension)
            action = action.clip(env.action_space.low[0],
                                 env.action_space.high[0])

        # act
        next_state, reward, done, observation = env.step(action)

        tot_reward += reward
        done = float(done)

        memory.add(state, next_state, action, reward, done)
        state = next_state

        if j >= start_exploiting:

            state_b, next_state_b, action_b, reward_b, done_b = memory.sample(memory_batch_size)
            state_b, next_state_b, action_b, reward_b, done_b = to_tensor(state_b, next_state_b, action_b, reward_b,
                                                                          done_b, device)

            with torch.no_grad():

                target_action = actor_target(next_state_b)
                target_action_noise = torch.clamp(torch.randn_like(action_b) * noise, -noise_clip,
                                                  noise_clip)
                target_action = torch.clamp(target_action + target_action_noise, env.action_space.low[0],
                                            env.action_space.high[0])

                c1, c2 = critic_target(next_state_b, target_action)
                min_target_q_value = reward_b + discount_factor * done_b * torch.min(c1, c2)

            q_value_1, q_value_2 = critic(state_b, action_b)

            critic_loss = F.mse_loss(q_value_1, min_target_q_value) + F.mse_loss(q_value_2, min_target_q_value)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if j % update_interval == 0:

                actor_loss = -critic.critic1(state_b, actor(state_b)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if done:
            # Reset environment
            state, done = env.reset(), False
            episode_number += 1
            print(
                f"Completed episode {episode_number} in {steps_in_episode} steps with a total reward of {tot_reward}    TOT: {step}")
            steps_in_episode = 0
            tot_reward = 0
            if episode_number % plot_interval == 0:

                reward_avg = 0

                for _ in range(evaluate_iterations):
                    state, done = env.reset(), False

                    while not done:
                        action = actor(torch.FloatTensor(state).to(device)).cpu().data.numpy()
                        state, reward, done, _ = env.step(action)
                        reward_avg += reward

                neptune.log_metric('avg_rewards', x=episode_number, y=reward_avg / evaluate_iterations)
                if reward_avg / evaluate_iterations > max_avg:
                    torch.save(actor.state_dict(), "_actor" + env_name + str(seed))
                    max_avg = max(max_avg, reward_avg / evaluate_iterations)
