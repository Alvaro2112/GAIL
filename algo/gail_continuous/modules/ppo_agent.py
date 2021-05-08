import torch
import torch.nn as nn

from modules.helper import ppo_iter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, gamma, clip, actor, critic, lr, update_batch_size, grad_clip,
                 entropy_weight, value_coef, betas):
        self.gamma = gamma
        self.clip = clip
        self.update_batch_size = update_batch_size
        self.grad_clip = grad_clip
        self.entropy_weight = entropy_weight
        self.value_coef = value_coef

        self.loss = nn.MSELoss()

        self.actor = actor
        self.critic = critic

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=betas)

    def update(self, ppo_updates, data):

        tot_loss = 0
        ob = data['states']
        ac = data['actions']
        oldpas = data['logpas']
        atarg = data['advants']
        tdlamret = data['tdlamret']
        vpredbefore = data['values']

        for i in range(ppo_updates):

            data_loader = ppo_iter(self.update_batch_size,
                                   ob, ac, oldpas, atarg, tdlamret, vpredbefore)

            for batch in data_loader:
                ob_b, ac_b, old_logpas_b, atarg_b, vtarg_b, old_vpred_b = batch

                # policy loss
                cur_logpas, cur_entropies = self.actor.get_predictions(ob_b, ac_b)
                ratio = torch.exp(cur_logpas - old_logpas_b)

                # clip ratio
                clipped_ratio = torch.clamp(ratio, 1. - self.clip, 1. + self.clip)

                # policy_loss
                surr1 = ratio * atarg_b.unsqueeze(1)

                surr2 = clipped_ratio * atarg_b.unsqueeze(1)
                pol_surr = -torch.min(surr1, surr2).mean()

                # value_loss
                cur_vpred = self.critic(ob_b).squeeze(1)

                # original value_loss
                vf_loss = (cur_vpred - vtarg_b).pow(2).mean()

                # entropy_loss
                pol_entpen = -cur_entropies.mean()

                # total loss
                c1 = self.value_coef
                c2 = self.entropy_weight

                # actor - backward
                self.actor_optimizer.zero_grad()
                policy_loss = pol_surr + c2 * pol_entpen
                policy_loss.backward()
                self.actor_optimizer.step()

                # critic - backward
                self.critic_optimizer.zero_grad()
                value_loss = c1 * vf_loss
                value_loss.backward()

                self.critic_optimizer.step()

                tot_loss += policy_loss

        return tot_loss / ppo_updates
