import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, env, net_dim, activation, max, min):
        super().__init__()

        self.net_dim = net_dim
        self.activation = activation
        self.state_dim = env.get_state_dim()
        self.max = max
        self.min = min

        self.actor1 = nn.Linear(self.state_dim, net_dim[0])
        self.actor2 = nn.Linear(net_dim[0], net_dim[1])
        self.actor3 = nn.Linear(net_dim[1], env.get_action_dim() + 1)


    def forward(self, state):
        x = torch.relu(self.actor1(state))
        x = torch.relu(self.actor2(x))
        policy_action = torch.sigmoid(self.actor3(x))
        return policy_action.cpu()

    def act(self, state):
        probabilities = self(state.to(device))
        distribution = Normal(probabilities[:-1], probabilities[-1])
        action = distribution.sample()
        action = action.clamp(-1, 1)
        log_probs = distribution.log_prob(action)
        return action, log_probs, distribution.entropy()

    def get_predictions(self, state, old_actions):
        probabilitiesx = self(state.to(device))
        std = []
        probabilities = []
        for i in probabilitiesx:
            probabilities.append(i.tolist()[:-1])
            std.append(i.tolist()[-1])
        std = torch.Tensor(std)
        probabilities = torch.Tensor(probabilities)

        distribution = Normal(probabilities, std.unsqueeze(1))
        log_probs = distribution.log_prob(old_actions)
        return log_probs, distribution.entropy()

