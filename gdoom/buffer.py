import numpy as np
import torch

GAMMA = 0.99 # reward discount factor (gamma), 1.0 = no discount
GAE_LAMBDA = 0.99

class Buffer:
    def __init__(self, batch_size, minibatch_size):
        self.states = []
        self.actions = []
        self.a_log_probs = []
        self.rewards_of_batch = []
        self.value_observations = []
        self.masks = []

        self.next_value = 0

        self.returns = []
        self.advantages = []

        self.batch_size = batch_size
        self.minibatch_size = minibatch_size

    def prepare_batch(self):
        self.returns = compute_gae(self.next_value, self.rewards_of_batch, self.masks, self.value_observations)
        self.states = torch.Tensor(self.states) / 255. # normalize images
        self.actions = torch.LongTensor(self.actions)
        # self.actions = self.actions.astype(np.int64)
        self.a_log_probs = torch.Tensor(self.a_log_probs)
        self.rewards_of_batch = torch.Tensor(self.rewards_of_batch)
        self.value_observations = torch.Tensor(self.value_observations)
        self.masks = torch.Tensor(self.masks)
        self.advantages = self.returns - self.value_observations
        self.normalize_advantages()

    def normalize_advantages(self):
        self.advantages -= self.advantages.mean()
        self.advantages /= (self.advantages.std() + 1e-8)

    def __iter__(self):
        index_order = np.arange(self.batch_size)
        np.random.shuffle(index_order)

        for minibatch_idx in range(self.batch_size // self.minibatch_size):
            random_indexes = index_order[minibatch_idx*self.minibatch_size:(minibatch_idx+1)*self.minibatch_size]
            print(minibatch_idx ,"/",self.batch_size // self.minibatch_size)
            yield (self.states[random_indexes],
                   self.actions[random_indexes],
                   self.a_log_probs[random_indexes],
                   self.advantages[random_indexes],
                   self.returns[random_indexes])



def compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        # prepend to get correct order back
        returns.insert(0, gae + values[step])
    return torch.Tensor(returns)
