import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, net_dim, lr, betas, expert, actor, update_batch_size):
        super().__init__()

        self.actor = actor
        self.expert = expert
        self.update_batch_size = update_batch_size
        self.exp_label = torch.full((update_batch_size, 1), 1, device=device)
        self.policy_label = torch.full((update_batch_size, 1), 0, device=device)

        self.discriminator1 = nn.Linear(state_dim + action_dim, net_dim[0])
        self.discriminator2 = nn.Linear(net_dim[0], net_dim[1])
        self.discriminator3 = nn.Linear(net_dim[1], 1)

        self.optimiser = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        self.loss_fn = nn.BCELoss()

    def forward(self, pair):
        x = torch.tanh(self.discriminator1(pair))
        x = torch.tanh(self.discriminator2(x))
        x = torch.sigmoid(self.discriminator3(x))

        return x

    def update(self):
        expert = self.expert.sample(self.update_batch_size).to(device)
        state = self.expert.sample_state(self.update_batch_size).to(device)
        action = self.actor(state)

        self.optimiser.zero_grad()
        prob_exp = self.forward(expert)
        loss = self.loss_fn(prob_exp.float(), self.exp_label.float())

        policy_action = self.actor(state)

        state_action = torch.cat([state, policy_action.detach()], 1)
        prob_policy = self.forward(state_action)

        loss += self.loss_fn(prob_policy.float(), self.policy_label.float())
        loss.backward()
        self.optimiser.step()

        state_action = torch.cat([state, action], 1)

        return -self.forward(state_action)
