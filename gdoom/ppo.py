from gym.utils.play import play
import numpy as np
from gdoom_env import *
from network import PolicyNet
from network import CriticNet
from train import train
import matplotlib

from matplotlib import pyplot as plt
import torch

class PPO:

    def __init__(self, scenario:str, pretrained=False):

        self.method = "ppo"
        self.env_string = "doom_" + scenario + "_640-v0"
        self.critic = CriticNet()
        self.env = gym.make(self.env_string)
        self.policy = PolicyNet(self.env.action_space.n)
        self.directory = 'stats/ppo/' + scenario + '/'
        if pretrained:
            print("Loading pretrained networks")
            self.policy = torch.load(self.directory+"policy_model")
            self.critic = torch.load(self.directory+"critic_model")
            self.critic.train()
            self.policy.train()
        print("GDoomEnv called")


##################
# Initialization #
##################
ppo = PPO("scenario4", pretrained=False)

################
# Train policy #
################

train(ppo)
