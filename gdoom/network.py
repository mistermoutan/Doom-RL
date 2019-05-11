import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import cv2
import utils
#import imageio
import torchvision.models as models

class PolicyNet(nn.Module):
    """Policy network"""

    def __init__(self,n_actions):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ELU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ELU())
        self.conv3 = nn.Sequential(
           nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
           nn.BatchNorm2d(128),
           nn.ELU())
        self.fc1 = nn.Sequential(
            nn.Linear(16*16*64, 512),
            nn.ELU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU())
        self.fc3 = nn.Sequential(
            nn.Linear(512, n_actions),
            nn.ELU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #print("End of conv ", x.shape)
        #x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #print(x)
        #print(nn.Softmax(dim=-1)(x))
        #print(nn.LogSoftmax(dim=-1)(x))
        return nn.LogSoftmax(dim=-1)(x)

class Critic_TL(nn.Module):
    def __init__(self,vgg):
        super(Critic_TL,self).__init__()
        self.final_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,1)
            )

        self.vgg = vgg
        for p in self.vgg.features.parameters():
            p.requires_grad=False
            
    def forward(self,x):
        x = self.vgg(x)
        x = self.final_layer(x)
        x *= 5
        return x


class CriticNet(nn.Module):
    """Critic network"""

    def __init__(self):
        super(CriticNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ELU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ELU())
        #self.conv3 = nn.Sequential(
        #    nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
        #    nn.BatchNorm2d(128),
        #    nn.ELU())
        self.fc1 = nn.Sequential(
            nn.Linear(16*16*64, 512),
            nn.ELU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU())
        self.fc3 = nn.Sequential(
            nn.Linear(256, 1))
            #nn.Sigmoid()) #benchmark


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        #x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
