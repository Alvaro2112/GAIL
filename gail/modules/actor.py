import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_dim, lr, betas):
        super().__init__()

        self.state_dim = state_dim
        self.layers = [None] * len(net_dim)

        self.actor1 = nn.Linear(state_dim, net_dim[0])
        self.actor2 = nn.Linear(net_dim[0], net_dim[1])
        #self.actor3 = nn.Linear(net_dim[1], net_dim[2])
        self.actor4 = nn.Linear(net_dim[1], action_dim)

        self.optimiser = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)

    def forward(self, state):
        x = torch.relu(self.actor1(state))
        x = torch.relu(self.actor2(x))
        #x = torch.relu(self.actor3(x))
        x = torch.tanh(self.actor4(x))

        return x

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self(state)[0].cpu().data.numpy().flatten()

    def update(self, loss):
        self.optimiser.zero_grad()
        loss.mean().backward()
        self.optimiser.step()
