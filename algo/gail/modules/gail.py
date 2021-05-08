import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GAIL:
    def __init__(self, actor, discriminator):
        self.actor = actor
        self.discriminator = discriminator

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state)[0].cpu().data.numpy().flatten()

    def update(self, n_iter):
        for i in range(n_iter):
            loss = self.discriminator.update()
            self.actor.update(loss)

        return self.actor.state_dict()
