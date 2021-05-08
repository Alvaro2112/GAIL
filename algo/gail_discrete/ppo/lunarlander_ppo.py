import gym
import neptune
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
neptune.init(project_qualified_name='alvaro/circuit-irl',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzg2YzgxMDQtNjMwOS00NjNlLTgzZTAtYmRiYmVlYWNiNWUzIn0=',
             )


class Neural(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Neural, self).__init__()

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

    def act(self, state, memory):
        state = torch.from_numpy(state).float()

        distribution = self.build_distribution(state)

        # Sample an action from distribution
        action = distribution.sample()

        # Store in memory
        memory.states.append(state)
        memory.actions.append(action)
        memory.log_probability.append(distribution.log_prob(action))

        # Returns value in tensor, if any
        return action.item()

    def evaluate(self, state, action):
        dist = self.build_distribution(state)
        state_value = self.value(state)
        action_logprobs = dist.log_prob(action)

        return action_logprobs, torch.squeeze(state_value)


class Replay:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probability = []
        self.rewards = []
        self.end = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.log_probability.clear()
        self.rewards.clear()
        self.end.clear()


class Agent:
    def __init__(self, state_space_dimension, action_space_dimension, gamma):

        self.gamma = gamma
        self.loss = nn.MSELoss()

        # Builds a new policy network
        self.policy = Neural(state_space_dimension, action_space_dimension).to(device=device)
        self.optimizer = torch.optim.Adam(self.policy.parameters())
        self.old_policy = Neural(state_space_dimension, action_space_dimension).to(device=device)

        # Copy new weigths into old policy
        self.old_policy.load_state_dict(self.policy.state_dict())

    def update(self, memory):

        rewards = []

        prev_states = torch.stack(memory.states).detach().to(device)
        prev_actions = torch.stack(memory.actions).detach().to(device)
        prev_log_probabilitys = torch.stack(memory.log_probability).detach()

        cummulative = 0

        # Compute discounted rewards
        for reward, end in zip(reversed(memory.rewards), reversed(memory.end)):
            cummulative = reward if end else reward + (self.gamma * cummulative)
            rewards.reverse()
            rewards.append(cummulative)
            rewards.reverse()

        rewards = torch.tensor(rewards)

        # Normalize
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for i in range(4):
            log_probability, values = self.policy.evaluate(prev_states, prev_actions)

            policy_ratio = torch.exp(log_probability - prev_log_probabilitys.detach())

            # Compute Objective function
            advantage = rewards - values.detach()
            loss = torch.min(policy_ratio * advantage,
                             torch.clamp(policy_ratio, 1 - self.clip, 1 + self.clip) * advantage) \
                   - self.loss(values, rewards)

            loss = -loss  # Maximize

            # Gradient descent
            self.optimizer.lr = self.learning_rate
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())

    def save(self, directory='./', name='lunarlander'):
        torch.save(self.policy.state_dict(), '{}/{}_actor.pth'.format(directory, name))


def main():
    # Environment Specifications
    env = gym.make("LunarLander-v2")
    state_dimension = env.observation_space.shape[0]
    action_dimension = env.action_space.n

    # Global variables
    max_reward = 230
    max_episodes = 5000
    max_steps = 1000
    update_period = 2000
    gamma = 0.999
    alpha = 1
    alpha_decrease = 1 / max_episodes
    ep_reward = 0
    last_x = max_episodes / 10  # Number of last episodes where the average reward has to be larger than max_reward

    # Initialize Memory and Agent
    replay_memory = Replay()
    agent = Agent(state_dimension, action_dimension, gamma)

    mean_reward, avgerage_steps, steps = 0, 0, 0
    rewards = []

    # Main loop (episodes)
    for i in range(max_episodes):

        # Update Agent learning parameters
        alpha -= alpha_decrease
        agent.clip = 0.1 * alpha
        agent.learning_rate = 2e-3 * alpha
        neptune.log_metric('learning_rate', agent.learning_rate)
        neptune.log_metric('alpha', alpha)

        state = env.reset()

        # Episode loop (steps)
        for no, j in enumerate(range(max_steps)):

            steps += 1

            # act
            action = agent.old_policy.act(state, replay_memory)
            state, reward, done, observation = env.step(action)

            # Memorize action and outcome
            replay_memory.rewards.append(reward)
            replay_memory.end.append(done)
            ep_reward += reward

            # Update in necessary
            if steps % update_period == 0:
                agent.update(replay_memory)
                replay_memory.clear()
                steps = 0

            mean_reward += reward / 20

            if done:
                break

        rewards.append(ep_reward)
        avgerage_steps += j / 20
        ep_reward = 0

        reward_average_last_x = np.array(rewards)
        reward_average_last_x = reward_average_last_x[-int(last_x):]

        # Check if mean of last last_x episodes is larger than required reward
        if np.sum(reward_average_last_x) / last_x >= max_reward:
            avgerage_steps = (avgerage_steps * 20) / ((i + 1) % 20)
            mean_reward = (mean_reward * 20) / ((i + 1) % 20)
            print('Episode number: {} \t average length: {} \t average reward: {}'.format((i + 1), int(avgerage_steps),
                                                                                          int(mean_reward)))
            print("Solved Environment")
            agent.save()
            break

        # Print info
        if (i + 1) % 20 == 0:
            print('Episode number: {} \t average length: {} \t average reward: {}'.format((i + 1), int(avgerage_steps),
                                                                                          int(mean_reward)))
            neptune.log_metric('avg_reward', y=int(mean_reward), x=i)
            mean_reward, avgerage_steps = 0, 0


if __name__ == '__main__':
    neptune.create_experiment()
    main()
