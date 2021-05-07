import numpy as np
import torch
import torch.nn as nn

from modules.helper import ppo_iter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Discriminator(nn.Module):
    def __init__(self, env, net_dim, lr, update_batch_size, activation, betas):
        super(Discriminator, self).__init__()

        self.l1 = nn.Linear(env.get_state_dim() + env.get_action_dim(), net_dim[0])
        self.l2 = nn.Linear(net_dim[0], net_dim[1])
        self.l3 = nn.Linear(net_dim[1], 1)

        self.optim_discriminator = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        self.loss_fn = nn.BCELoss()
        self.update_batch_size = update_batch_size
        self.action_dim = env.get_action_dim()
        self.activation = activation
        self.init_pointer = 0

    def forward(self, state):

        x = torch.nn.LeakyReLU(0.2)(self.l1(state.to(device)))
        x = torch.nn.LeakyReLU(0.2)(self.l2(x))
        x = torch.sigmoid(self.l3(x))  # remove sigmoid. TODO

        return x

    def get_next_batch(self, pool):
        if self.init_pointer + self.update_batch_size > len(pool):
            self.init_pointer = 0
            np.random.shuffle(np.arange(len(pool)))

        # get demo states and actions
        demo_states = [i[0] for i in pool][self.init_pointer: self.init_pointer + self.update_batch_size]
        demo_actions = [i[1] for i in pool][self.init_pointer: self.init_pointer + self.update_batch_size]

        self.init_pointer += self.update_batch_size

        return demo_states, demo_actions

    def update(self, pool, data):
        learner_ob = data['states']
        learner_ac = data['actions']

        rub = torch.zeros_like(learner_ob)
        learner_iter = ppo_iter(self.update_batch_size, learner_ob, learner_ac, rub, rub, rub, rub)

        for learner_ob_b, learner_ac_b, _, _, _, _ in learner_iter:
            expert_ob_b, expert_ac_b = self.get_next_batch(pool)

            trajs = torch.cat([learner_ob_b, learner_ac_b.float()], 1)
            learner_prob = self(trajs)

            expert_trajs = torch.cat(
                [torch.FloatTensor(expert_ob_b).squeeze(1), torch.FloatTensor(expert_ac_b)], 1)
            expert_prob = self(expert_trajs)

            learner_loss = self.loss_fn(learner_prob, torch.ones_like(learner_prob))
            expert_loss = self.loss_fn(expert_prob, torch.zeros_like(expert_prob))

            loss = learner_loss + expert_loss

            self.optim_discriminator.zero_grad()
            loss.backward()
            self.optim_discriminator.step()

        return loss
