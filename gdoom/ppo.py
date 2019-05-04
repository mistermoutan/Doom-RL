from gym.utils.play import play
import numpy as np
from gdoom_env import *
from network import PolicyNet
from network import CriticNet
from network import Critic_TL
import torchvision.models as models
from train import train
import torch.nn as nn
import matplotlib

from matplotlib import pyplot as plt
import torch

class Softmax(nn.Module):
    def __init__(self, in_dim=1000,):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features=in_dim, out_features=256),
                                 nn.LogSoftmax(dim=-1))
        
    def forward(self,x):        
        return self.net(x)
class PPO:

    def __init__(self,env:str,pre_trained = False):

        self.method = "ppo"
        self.env_string = env
        self.env = gym.make(env)
        self.pre_trained = pre_trained
        if pre_trained:
            model = models.vgg11(pretrained=pre_trained)
            model.classifier[6] = nn.Linear(4096,self.env.action_space.n)
            self.policy = model
            model_critc = models.vgg11(pretrained=pre_trained)
            self.critic = Critic_TL(model_critc)
        else:
            self.policy = PolicyNet(self.env.action_space.n)
            self.critic = CriticNet()

        print("GDoomEnv called")


##################
# Initialization #
##################
ppo = PPO("doom_scenario0_640-v0",pre_trained=True)

################
# Train policy #
################

train(ppo)
