import neptune
import numpy as np
import torch
import torch.nn as nn


def get_reward(observation, action, discriminator):
    observation = torch.FloatTensor(np.expand_dims(observation, 0)).squeeze(0)
    traj = torch.cat([observation, action], 0)

    reward = discriminator(traj).detach()
    reward = - torch.log(reward)

    return reward.item()


def ppo_iter(batch_size, ob, ac, oldpas, atarg, tdlamret, vpredbefore):
    total_size = ob.size(0)
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    n_batches = total_size // batch_size
    for nb in range(n_batches):
        ind = indices[batch_size * nb: batch_size * (nb + 1)]
        yield ob[ind], torch.tensor(ac)[ind], torch.tensor(oldpas)[ind], atarg[ind], tdlamret[ind], vpredbefore[ind]


def init_normal_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


def prepare_data(data):
    states_tensor = torch.from_numpy(np.stack(data['states'])).float()
    actions_tensor = data['actions']
    logpas_tensor = data['logpas']
    tdlamret_tensor = torch.tensor(data['tdlamret']).float()
    advants_tensor = torch.tensor(data['advants']).float()
    values_tensor = torch.tensor(data['values']).float()

    advants_tensor = (advants_tensor - advants_tensor.mean()) / (advants_tensor.std() + 1e-5)

    data_tensor = dict(states=states_tensor, actions=actions_tensor, logpas=logpas_tensor,
                       tdlamret=tdlamret_tensor, advants=advants_tensor, values=values_tensor)

    return data_tensor


def log_metrics(evaluation_interval, tot_reward, epoch, tot_policy_loss, tot_discr_loss, episodes_in_stats, record):
    if episodes_in_stats == 0:
        avg_reward = 0
    else:
        avg_reward = tot_reward / episodes_in_stats
    print("Epoch {} with {}".format(epoch, avg_reward))
    if record:
        neptune.log_metric('reward', y=avg_reward, x=epoch)
        neptune.log_metric('policy_loss', y=tot_policy_loss / evaluation_interval, x=epoch)
        neptune.log_metric('discriminator_loss', y=tot_discr_loss / evaluation_interval, x=epoch)


def initiate_run(env_name, actor_net_dim, critic_net_dim, dsicriminator_net_dim,
                 lr, gamma, tau, grad_clip, batch_size, entropy_weight,
                 min_buffer_size, clip, ppo_updates, discriminator_updates, expert, activation, value_coef, betas,
                 max_steps, seed, tag, record):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if record:
        neptune.create_experiment(
            params={'lr': lr, 'gamma': gamma, 'lambda': tau, 'grad_clip': grad_clip, 'entropy_weight': entropy_weight,
                    'value_coef': value_coef,
                    'bsize': batch_size, 'clip': clip, 'ppo_updates': ppo_updates, 'max_steps': max_steps,
                    'betas[0]': betas[0], 'betas[1]': betas[1],
                    'discriminator_updates': discriminator_updates,
                    'min_buffer_size': min_buffer_size,
                    'activation': activation, 'actor_net_dim[0]': actor_net_dim[0],
                    'actor_net_dim[1]': actor_net_dim[1],
                    'critic_net_dim[0]': critic_net_dim[0], 'critic_net_dim[1]': critic_net_dim[1],
                    'dsicriminator_net_dim[0]': dsicriminator_net_dim[0],
                    'dsicriminator_net_dim[1]': dsicriminator_net_dim[1], "traj": expert,
                    "seed": seed}, tags=[env_name + "#" + str(expert) + '#' + str(seed) + "#" + tag])

    if activation == "tanh":
        activation = torch.tanh
    if activation == "relu":
        activation = torch.relu
    if activation == "swish":
        def activation(): var = lambda x: x * torch.nn.functional.sigmoid(x)
    if activation == "mish":
        def activation(): var = lambda x: x * (torch.tanh(torch.nn.functional.softplus(x)))

    if env_name is not None:
        expert = './ppo/policies/cheetah' + '.pkl'

    return expert, activation
