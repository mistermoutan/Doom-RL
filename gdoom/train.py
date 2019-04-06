import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import utils
import imageio
imageio.plugins.ffmpeg.download()
import copy
from buffer import Buffer

"""
class Model:

    def __init__(self,policy,cric)
"""

lr_actor = 1e-5
lr_critic = 1e-4
num_epochs = 500
batch_size = 256
minibatch_size = 64

PPO_EPSILON = 0.2
CRITIC_BALANCE = 0.3
ENTROPY_BETA = 0.001

cuda = torch.cuda.is_available()
device = "cuda:0" if cuda else "cpu"

def train(env, policy,critic):

    optimizer_actor = optim.Adam(policy.parameters(), lr=lr_actor) #update parameters
    optimizer_critic = optim.Adam(critic.parameters(),lr = lr_critic) #update parameters

    policy = policy.to(device)
    critic = critic.to(device)

    loss_lsq = torch.nn.MSELoss()
    NLL = nn.NLLLoss(reduction='none') # cost

    training_rewards, losses_actor , losses_critic = [], [], []
    print('Start training')

    for epoch in range(num_epochs):

        #first initialize env

        s = env.reset()
        states_human_size = [s] #keep 640*640 frame for display purpose
        s = cropping(s)
        num_episode = 1

        #create empty buffer
        batch_buffer = Buffer(batch_size, minibatch_size)
        rewards_of_episode = []

        # build batch
        for idx_in_batch in range(batch_size):
            # generate rollout by iteratively evaluating the current policy on the environment
            with torch.no_grad():

                s_tensor = torch.from_numpy(normalize(s)).float().permute(2,0,1).view(1,4,64,64)
                s_tensor = s_tensor.to(device)
                a_log_probs = policy(s_tensor)  #calls forward function
                estimated_value = critic(s_tensor)

            #print(a_prob.numpy())

            a = (np.cumsum(np.exp(a_log_probs.cpu().numpy())) > np.random.rand()).argmax() # sample action
            s1, r, done, info = env.step(int(a)) #s1 comes in 640x640x4
            states_human_size.append(np.asarray(s1))
            s1 = cropping(s1)
            batch_buffer.states.append(s)
            batch_buffer.actions.append(a)
            batch_buffer.a_log_probs.append(a_log_probs.cpu().numpy())
            batch_buffer.rewards_of_batch.append(r)
            batch_buffer.value_observations.append(estimated_value) #y's of critic

            if not done:
                batch_buffer.masks.append(1)

            if done:
                batch_buffer.masks.append(0)
                rewards_of_episode = []
                num_episode += 1
                s = cropping(env.reset())
            else:
                s = s1

        s_tensor = torch.from_numpy(normalize(s)).float().permute(2,0,1).view(1,4,64,64)
        s_tensor = s_tensor.to(device)
        next_value = critic(s_tensor)
        batch_buffer.next_value = next_value
        # prepare batch
        batch_buffer.prepare_batch()

        for states, actions, log_prob_old, advantages, returns in batch_buffer:
            states = states.to(device)
            a_log_probs = policy(states.permute(0,3,1,2)) # permute because of channel first in pytorch conv layer
            values = critic(states.permute(0,3,1,2))
            print(actions)
            log_likelihood_new = NLL(a_log_probs, torch.LongTensor(actions).to(device))
            log_likelihood_old = NLL(log_prob_old.view(minibatch_size, -1).to(device), torch.LongTensor(actions).to(device))

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()

            advantages = Variable(advantages, requires_grad=True).to(device)

            # A2C losses
            # loss_actor = torch.sum(advantages.detach()*(log_likelihood)) #mean vs sum
            # loss_critic = torch.sum(advantages.pow(2) )

            #PPO losses
            prob_ratio = torch.exp(log_likelihood_old - log_likelihood_new) #opposite sign cause computed by NLL
            surrogate_objective = prob_ratio * advantages
            surrogate_objective_clipped = torch.clamp(prob_ratio, 0.98 - PPO_EPSILON, 1.02 + PPO_EPSILON) * advantages

            entropy = torch.exp(a_log_probs) * a_log_probs

            loss_actor = - torch.min(surrogate_objective, surrogate_objective_clipped).mean()
            advantages_new = returns.to(device) - values.view(-1)
            loss_critic = advantages_new.pow(2).mean()

            loss = loss_actor + CRITIC_BALANCE * loss_critic + ENTROPY_BETA * entropy.mean()

            # loss_actor.backward()
            # loss_critic.backward()

            loss.backward()

            optimizer_actor.step()
            optimizer_critic.step()


            losses_actor.append(loss_actor.item())
            losses_critic.append(loss_critic.item())



           # bookkeeping

        print("==========================================")
        print("Epoch: ", epoch, "/", num_epochs)
        print("-----------")
        print("Number of training episodes: {}".format(num_episode))
        print("Total reward: {0:.2f}".format(batch_buffer.rewards_of_batch.sum()))
        print("Mean Reward of that batch {0:.2f}".format(batch_buffer.rewards_of_batch.mean()))
        print("Training Loss for Actor: {0:.2f}".format(loss_actor.item()))
        print("Training Loss for Critic: {0:.2f}".format(loss_critic.item()))
        #print("Length of last episode: {0:.2f}".format(rewards_of_batch.shape[0]))


        if (epoch+1) % 10 == 0:
            format_frames = np.array(states_human_size)
            imageio.mimwrite('videos/training_ppo'+str(epoch+1)+'.mp4', format_frames[:,:,:,0], fps = 15)



    print('done')


def normalize(img):
    return (img)/255.0

def cropping(s):
    frame = np.asarray(s)
    frame = cv2.resize(frame, (64,64))
    return frame
