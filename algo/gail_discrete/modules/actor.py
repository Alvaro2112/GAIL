import torch
import torch.nn as nn
from torch.distributions import Categorical


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, env, net_dim, activation):
        super().__init__()

        self.net_dim = net_dim
        self.activation = activation
        self.state_dim = env.get_state_dim()

        self.actor1 = nn.Linear(self.state_dim, net_dim[0])
        self.actor2 = nn.Linear(net_dim[0], net_dim[1])
        self.actor3 = nn.Linear(net_dim[1], env.get_action_dim())


    def forward(self, state):
        x = torch.relu(self.actor1(state))
        x = torch.relu(self.actor2(x))
        policy_action = torch.sigmoid(self.actor3(x))
        return policy_action.cpu()

    def act(self, state):
        probabilitys = self(state.to(device))
        distribution = Categorical(probabilitys)
        action = distribution.sample()
        log_probs = distribution.log_prob(action)
        return action.item(), log_probs, distribution.entropy()

    def get_predictions(self, state, old_actions):
        probabilitys = self(state.to(device))
        distribution = Categorical(probabilitys)
        log_probs = distribution.log_prob(old_actions)
        return log_probs, distribution.entropy()

    def greedy_act(self, state):
        return self(state).argmax(0).item()
