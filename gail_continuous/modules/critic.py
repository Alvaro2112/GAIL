import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    def __init__(self, env, net_dim, activation):
        super().__init__()

        self.fc1 = nn.Linear(env.get_state_dim(), net_dim[0])
        self.fc2 = nn.Linear(net_dim[0], net_dim[1])
        self.fc3 = nn.Linear(net_dim[1], 1)

        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc1(x.to(device)))
        x = self.activation(self.fc2(x))
        v = self.fc3(x)

        return v.cpu()
