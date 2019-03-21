print("GDoomEnv called")
from gym.utils.play import play
import numpy as np
from gdoom_env import *
from network import PolicyNet
from train import train
import matplotlib
import matplotlib.pyplot as plt
import torch




##################
# Initialization #
##################

# Make a CPU environemnt the good ol' way (not recommended, see __init__.py).
genv = WGDoomEnv(level=1, frame_size=89)
genv.reset()
a, _, _, _ = genv.step(0)
print( np.asarray(a).shape )

# Also make a GPU environment, but using openai:
env = gym.make("doom_scenario4_640-v0")
frame = env.reset()

print("Frame size for cpu player: ", np.asarray(frame).shape )



# env_human = gym.make("doom_scenario2_human-v0")
# frame = env_human.reset()
# print("Frame size for homan player: ", np.asarray(frame).shape)

####################
# Creating network #
####################

n_inputs = env.observation_space.shape[0]**2*4
print(n_inputs)
n_hidden = 256
learning_rate = 0.00005
n_outputs = env.action_space.n

policy = PolicyNet()
print(policy)
print(policy(torch.randn(1,4,64,64)))


################
# Train policy #
################
train(env, policy)
