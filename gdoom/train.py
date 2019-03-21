import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import imageio

import cv2
import utils

def train(env, policy):

    num_epochs = 2000
    batch_size = 10 # max rollout length
    discount_factor = 0.95 # reward discount factor (gamma), 1.0 = no discount
    learning_rate = 0.00005

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    NLL = nn.NLLLoss(reduction='none')

    training_rewards, losses = [], []
    print('Start training')
    for epoch in range(num_epochs):
        batch = []
        s = env.reset()
        states_human_size = [s]
        s = cropping(s)
        num_episode = 1
        states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []

        while True:
            # generate rollout by iteratively evaluating the current policy on the environment
            with torch.no_grad():
                s_tensor = torch.from_numpy(normalize(s)).float().permute(2,0,1).view(1,4,64,64)
                # print(s_tensor.shape)
                a_prob = policy(s_tensor)

            a = (np.cumsum(np.exp(a_prob.numpy())) > np.random.rand()).argmax() # sample action
            s1, r, done, info = env.step(int(a))
            states_human_size.append(np.asarray(s1))
            s1 = cropping(s1)
            states.append(s)

            actions.append(a)
            rewards_of_episode.append(r)

            if done:
                rewards_of_batch.append(rewards_of_episode)
                discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode, discount_factor))
                if len(np.concatenate(rewards_of_batch)) > batch_size:
                    break

                rewards_of_episode = []
                num_episode += 1

                s = cropping(env.reset())
            else:
                s = s1

        # prepare batch
        states = np.stack(np.array(states))
        actions = np.stack(np.array(actions))
        actions = actions.astype(np.int64)
        rewards_of_batch = np.concatenate(rewards_of_batch)
        returns = np.concatenate(discounted_rewards)


        # policy gradient update
        actual_batch_size = states.shape[0] #can be different from batch size as episode length is not constant
        optimizer.zero_grad()
        states_batch = torch.from_numpy(normalize(states)).float().permute(0,3,2,1)
        # print(states_batch.shape)
        a_probs = policy(states_batch)
        # print(a_probs)
        d_rewards = torch.Tensor(returns).view(-1)
        # print(d_rewards)
        likelihood = NLL(a_probs, torch.LongTensor(actions))
        print(likelihood)
        loss = torch.sum(d_rewards * likelihood)
        loss.backward()

        optimizer.step()

        losses.append(loss.item())
        # bookkeeping
        training_rewards.append(np.mean(rewards_of_batch))

        print("==========================================")
        print("Epoch: ", epoch, "/", num_epochs)
        print("-----------")
        print("Number of training episodes: {}".format(num_episode))
        print("Length of last episode: {}".format(rewards_of_batch.shape[0]))
        print("Total reward: {}".format(np.sum(rewards_of_batch)))
        print("Mean Reward of that batch {}".format(np.mean(rewards_of_batch)))
        print("Training Loss: {}".format(loss.item()))
        if (epoch+1) % 50 == 0:
            format_frames = np.array(states_human_size)
            imageio.mimwrite('videos/training_'+str(epoch+1)+'.mp4', format_frames[:,:,:,0], fps = 15)



    print('done')

def discount_and_normalize_rewards(episode_rewards, discount_factor):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * discount_factor + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    # print(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    # print(discounted_episode_rewards)
    return discounted_episode_rewards


def normalize(img):
    return (img)/255.0

def cropping(s):
    frame = np.asarray(s)
    frame = cv2.resize(frame, (64,64))
    return frame

def display_episode(frames):
    img = None
    for frame in frames[:,:,:,0]:
        if img==None:
            img = plt.imshow(np.asarray(frame), cmap='gray')
        else:
            img.set_data(np.asarray(frame))
        plt.pause(0.1)
        plt.draw()
