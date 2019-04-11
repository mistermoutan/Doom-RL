import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from math import exp
from random import random, randrange
from itertools import count


import cv2
import utils

from experienceReplay import *
from net import *

## USE THIS
# https://gist.github.com/simoninithomas/7611db5d8a6f3edde269e18b97fa4d0c#file-deep-q-learning-with-doom-ipynb
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#########################################################################################################
#########################################################################################################
#########################################################################################################

def torchify(state):
    return torch.from_numpy(state).float().permute(2,0,1).view(1,4,64,64)

#########################################################################################################
#########################################################################################################
#########################################################################################################

# Hyperparameters

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

class Trainer:

    def __init__(self, env, device, n_actions, memory):
        self.env = env
        self.device = device
        self.n_actions = n_actions

        self.policy_net = DeepQNet(nbrActions=n_actions).to(device)
        self.target_net = DeepQNet(nbrActions=n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = memory

        self.steps_done = 0
        self.episode_durations = []

        self.display_every = 1

    def getMemory(self):
        return self.memory

    def setMemory(self, memory):
        self.memory = memory
        return

    def getPolicyNet(self):
        return self.policy_net

    def setPolicyNet(self, policy_net):
        self.policy_net = policy_net
        return

    def getTargetNet(self):
        return self.target_net

    def setTargetNet(self, target_net):
        self.target_net = target_net
        return

    def select_action(self, state):
        sample = random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):

        if len(self.memory) < BATCH_SIZE:
            return

        print('Start Optimization')
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        print('Start Backward')
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        print('End Optimization')


    def train(self, num_episodes=2):

        for i_episode in range(num_episodes):
            print('Episode: {0}'.format(i_episode + 1))

            # Initialize the environment and state
            state  = self.env.reset()
            states_human_size = [np.asarray(state)] # Allow visualization of episode.
            state = torchify(preprocessState(state))
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action.item())
                states_human_size.append(np.asarray(next_state))
                reward = torch.tensor([reward], device=self.device)
            
                # Observe new state
                if not done:
                    next_state = torchify(preprocessState(next_state))
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    # plot_durations() # TODO
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if (i_episode+1) % self.display_every == 0:
                display_episode(np.array(states_human_size))
