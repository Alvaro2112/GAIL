import argparse
from typing import Tuple


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--env', type=str)
    parser.add_argument('--actor_net_dim', type=Tuple[int, ...])
    parser.add_argument('--discriminator_net_dim', type=Tuple[int, ...])
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--eval_frequency', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--betas', type=Tuple[int, ...])
    parser.add_argument('--n_iter', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--n_eval_episodes', type=int)
    parser.add_argument('--expert', type=str)
    args = parser.parse_args()

    return args
