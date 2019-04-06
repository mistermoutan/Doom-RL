import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import matplotlib
matplotlib.use('TkAgg')




class ActorCriticNet(nn.Module):
    """Policy network"""

    def __init__(self,n_actions):
        super(ActorCriticNet, self).__init__()

        self.n_actions = n_actions

        cuda = torch.cuda.is_available()
        self.device = "cuda:0" if cuda else "cpu"

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv3 = nn.Sequential(
           nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
           nn.BatchNorm2d(64),
           nn.ReLU())

        self.actor = nn.Sequential(nn.Linear(8*8*64, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, n_actions))

        self.critic = nn.Sequential(nn.Linear(8*8*64, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1))



    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(batch_size, -1)

        probs = self.actor(x)
        probs = nn.Softmax(dim=-1)(probs)
        dist = Categorical(probs)

        value = self.critic(x)
        mask_log_prob = torch.Tensor(np.arange(self.n_actions)).repeat(batch_size,1).to(self.device)
        mask_log_prob = mask_log_prob.permute(1,0)
        return dist.log_prob(mask_log_prob).permute(1,0), value, dist
