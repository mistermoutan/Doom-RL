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
env = gym.make("doom_scenario2_640-v0")
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
memorySize = 100000
memory = ReplayMemory(memorySize)

# Initialize memory with 1000 random states
trainer = Trainer(env, device, n_actions, memory)

################
# Train policy #
################
trainer.preTrainMemory(pre_train=int(memorySize/100))
print('---Done Pre Training---')
trainer.train(num_episodes=1000)

# Saving
RECOVER_MEMORY = TRUE
if RECOVER_MEMORY:
    memory = readPickled(HOME_DIR + '/saves/memory{0}.pickle'.format(memorySize))
else:
    memory = trainer.getMemory()
    serializeObject(memory, HOME_DIR + '/saves/memory{0}.pickle'.format(memorySize))


policyNet = trainer.getPolicyNet()
serializeObject(policyNet, HOME_DIR + '/saves/policyNet.pickle')
targetNet = trainer.getTargetNet()
serializeObject(targetNet, HOME_DIR + '/saves/targetNet.pickle')
