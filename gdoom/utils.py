import sys, os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from IPython.display import clear_output


from math import exp
from statistics import mean
from random import random, randrange, choices
from itertools import count

import cv2

import warnings
warnings.filterwarnings("ignore")

def blockPrint():
    '''
    Disable print() function
    '''
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    '''
    Enable print() function
    '''
    sys.stdout = sys.__stdout__


def torchify(state, device):
    '''
    Allow to transform states into usable tensors.
    '''
    return torch.from_numpy(state).float().permute(2,0,1).view(1,4,64,64).to(device)

def preprocessState(state):
    '''
    Preprocess state (=stack of 4 frames)
    Turns frames into 64*64 frames.
    '''
    state = np.asarray(state)
    state = cv2.resize(state, (64,64))
    return (state)/255.0

def display_episode(frames):
    '''
    DISPLAY an episode from recorded frames.
    '''
    img = None
    for frame in frames[:,:,:,0]:
        if img==None:
            img = plt.imshow(np.asarray(frame), cmap='gray')
        else:
            img.set_data(np.asarray(frame))
        plt.pause(0.1)
        plt.draw()

def random_policy_human():
    ###########################################################
    # Should not work like this, need to give some parameters #
    ###########################################################

    # cv2.imshow("image",np.asarray(frame))
    img = None
    for i in range(500):
        if img==None:
            img = plt.imshow(np.asarray(frame))
        else:
            img.set_data(np.asarray(frame))
        plt.pause(0.01)
        plt.draw()
        # This will just create a sample action in any environment.
        #frame In this environment, the action can be 0 or 1, which is left or right
        action = env_human.action_space.sample()

        # this executes the environment with an action,
        # and returns the observation of the environment,
        # the reward, if the env is over, and other info.
        frame, _, _, _ =env_human.step(action)

def plotRewardsLosses(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(122)
    plt.title('loss')
    plt.plot(losses)
    plt.show()
