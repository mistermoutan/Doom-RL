import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib
matplotlib.use('TKAgg')   
import matplotlib.pyplot as plt
import cv2
import utils
import imageio

def train(env, policy,critic):

    num_epochs = 2000
    batch_size = 1 # max rollout length
    discount_factor = 0.95 # reward discount factor (gamma), 1.0 = no discount
    learning_rate = 0.0005
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate) #update parameters
    optimizer_critic = optim.Adam(critic.parameters(),lr = learning_rate) #update parameters
    loss_fn = torch.nn.MSELoss()
    NLL = nn.NLLLoss(reduction='none') # cost 

    #optimiser = optim.Adam([
    #    {'params': policy.fc.parameters(), 'lr': 0.01},
    #], lr=0)
    

    training_rewards, losses = [], []
    print('Start training')
    for epoch in range(num_epochs):
        batch = []
        s = env.reset()
        states_human_size = [s]
        s = cropping(s)
        num_episode = 1
        states, actions, rewards_of_episode, rewards_of_batch, discounted_rewards = [], [], [], [], []
        value_observations = []

        # build batch
        while True:
            # generate rollout by iteratively evaluating the current policy on the environment
            with torch.no_grad():

                s_tensor = torch.from_numpy(normalize(s)).float().permute(2,0,1).view(1,4,64,64)
                a_prob = policy(s_tensor)  #calls forward function
                estimated_value = critic(s_tensor)
            
            #a_prob_np = a_prob.numpy()   

            a = (np.cumsum(np.exp(a_prob.numpy())) > np.random.rand()).argmax() # sample action
            s1, r, done, info = env.step(int(a)) #s1 comes in 640x640x4
            states_human_size.append(np.asarray(s1))
            s1 = cropping(s1)
            states.append(s)
            actions.append(a)
            rewards_of_episode.append(r)
            value_observations.append(estimated_value) #y's of critic

            if done:
                rewards_of_batch.append(rewards_of_episode)
                # isn't this just making the final rewards of an episode go small, this should be for each time step
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

        #observed_values_critic = np.stack(np.array(value_observations))
        observed_values_critic = np.concatenate(value_observations)
        target_values_critic = critic_target_values(rewards_of_batch)
        target_values_critic = np.concatenate(target_values_critic)
        rewards_of_batch = np.concatenate(rewards_of_batch)
        returns = np.concatenate(discounted_rewards)

        #update critic
        target = torch.Tensor(target_values_critic).view(-1)
        observations = torch.Tensor(observed_values_critic).view(-1) 
        target = Variable(target,requires_grad = True)
        observations = Variable(observations,requires_grad = True)
        loss_critic = loss_fn(observations, target)  
        print("loss\n\n", loss_critic)
        #loss_critic = Variable(loss_critic, requires_grad = True)
        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()

        # V of st+1, the first value of batch will be innacurate
        observed_values_critic_shifted = np.roll(observed_values_critic,-1)


        # policy gradient update
        actual_batch_size = states.shape[0] #can be different from batch size as episode length is not constant
        optimizer.zero_grad()
        states_batch = torch.from_numpy(normalize(states)).float().permute(0,3,2,1)
        a_probs = policy(states_batch)
        #print(rewards_of_batch.shape,observed_values_critic_shifted.shape,observed_values_critic.shape)
        observed_values_critic_shifted = observed_values_critic_shifted.reshape(-1)
        observed_values_critic = observed_values_critic.reshape(-1)
        advantages = rewards_of_batch + observed_values_critic_shifted - observed_values_critic
        print("shape pre advantages:",advantages.shape)
        advantages_tensor = torch.Tensor(advantages).view(-1)
        
        print("advantages: {0} \n a_probs : {1} \n actions: {2}".format(advantages_tensor.shape,a_probs.shape,actions.shape))
        #d_rewards = torch.Tensor(returns).view(-1)
        #print("a_probs",a_probs)
        #print(states_batch.shape)
        #print(d_rewards)

        loss = torch.mean(advantages_tensor*NLL(a_probs, torch.LongTensor(actions)))  #thorch.mean vs torch.sum
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
        if (epoch+1) % 5 == 0:
            display_episode(np.array(states_human_size))
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

def critic_target_values(batch_rewards):

    # at each state, the target value is the sum of the rewards that were still obtained in the episode
    for episode in batch_rewards:
        for episode_reward in (range(len(episode))):

            episode[episode_reward] = sum(episode[episode_reward:])

    return batch_rewards
            


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
        plt.pause(0.00001)
        plt.draw()
