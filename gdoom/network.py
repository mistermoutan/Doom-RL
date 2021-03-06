import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNet(nn.Module):
    """Policy network"""

    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ELU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ELU())
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(128),
        #     nn.ELU())
        self.fc1 = nn.Sequential(
            nn.Linear(16*16*64, 512),
            nn.ELU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, 3),
            nn.ELU())
        # self.fc3 = nn.Sequential(
        #     nn.Linear(512, 3),
        #     nn.ELU())



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        # print(x)
        return nn.LogSoftmax(dim=-1)(x)
