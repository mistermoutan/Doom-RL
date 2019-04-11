from gym.utils.play import play
import numpy as np
from gdoom_env import *
from train import *
from experienceReplay import *
import matplotlib
import torch.optim as optim


import matplotlib.pyplot as plt
import torch


#########################################################################################################
#########################################################################################################
#########################################################################################################

# Make a CPU environemnt the good ol' way (not recommended, see __init__.py).
genv = WGDoomEnv(level=1, frame_size=640)

# Also make a GPU environment, but using openai:
env = gym.make("doom_scenario4_640-v0")
frame = env.reset()

#########################################################################################################
#########################################################################################################
#########################################################################################################

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################
# Creating network #
####################

# Get number of actions from gym action space
n_actions = env.action_space.n

memory = ReplayMemory(10000)

trainer = Trainer(env, device, n_actions, memory)

################
# Train policy #
################
trainer.train()

# Saving
memory = trainer.getMemory()
serializeObject(memory, HOME_DIR + '/saves/memory10000.pickle')
policyNet = trainer.getPolicyNet()
serializeObject(policyNet, HOME_DIR + '/saves/policyNet.pickle')
targetNet = trainer.getTargetNet()
serializeObject(targetNet, HOME_DIR + '/saves/targetNet.pickle')
