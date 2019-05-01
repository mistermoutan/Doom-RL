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

    def __init__(self,env:str):

        self.method = "ppo"
        self.env_string = env
        self.critic = CriticNet()
        self.env = gym.make(env)
        self.policy = PolicyNet(self.env.action_space.n)
        print("GDoomEnv called")


##################
# Initialization #
##################
ppo = PPO("doom_scenario2_640-v0")

################
# Train policy #
################

train(ppo)
