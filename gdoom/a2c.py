from gym.utils.play import play
import numpy as np
from gdoom_env import *
from network import PolicyNet 
from network import CriticNet 
from train import train_a2c
import matplotlib

from matplotlib import pyplot as plt
import torch

class A2C:

    def __init__(self,env:str):

        self.method = "a2c"
        self.env_string = env
        self.critic = CriticNet()
        self.env = gym.make(env)
        self.policy = PolicyNet(self.env.action_space.n)
        print("GDoomEnv called")


##################
# Initialization #
##################
a2c = A2C("doom_scenario2_640-v0")

################
# Train policy #
################

train_a2c(a2c)
