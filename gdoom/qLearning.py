from gym.utils.play import play
import numpy as np
from gdoom_env import *
from net import Net3Actions
from train import train
import matplotlib

import matplotlib.pyplot as plt
import torch


#########################################################################################################
#########################################################################################################
#########################################################################################################

# Make a CPU environemnt the good ol' way (not recommended, see __init__.py).
genv = WGDoomEnv(level=1, frame_size=640)
genv.reset()
a, _, _, _ = genv.step(0)
print( np.asarray(a).shape )

# Also make a GPU environment, but using openai:
env = gym.make("doom_scenario4_640-v0")
frame = env.reset()

print("Frame size for cpu player: ", np.asarray(frame).shape )

#########################################################################################################
#########################################################################################################
#########################################################################################################

####################
# Creating network #
####################

net = Net3Actions()
print(net)
print(net(torch.randn(64*64*4))) # test with random input


################
# Train policy #
################
train(env, net)
