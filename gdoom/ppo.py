from gym.utils.play import play
import numpy as np
from gdoom_env import *
from network import PolicyNet
from network import CriticNet
import torchvision.models as models
from train import train
import torch.nn as nn
import matplotlib

from matplotlib import pyplot as plt
import torch

class PPO:

    def __init__(self,env:str,pre_trained = False):

        self.method = "ppo"
        self.env_string = env
        self.critic = CriticNet()
        self.env = gym.make(env)
        if pre_trained:
            model = models.vgg11(pretrained=pre_trained)
            model.classifier[6] = nn.Linear(4096,self.env.action_space.n)
            self.policy = model
        else:
            self.policy = PolicyNet(self.env.action_space.n)
        self.pre_trained = pre_trained
        print("GDoomEnv called")


##################
# Initialization #
##################
ppo = PPO("doom_scenario1_640-v0",pre_trained=True)

################
# Train policy #
################

train(ppo)
