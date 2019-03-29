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
import copy

"""
class Model:

    def __init__(self,policy,cric)
"""
def train(env, policy,critic):

    num_epochs = 2000
    batch_size = 100 # max rollout length
    discount_factor = 0.95 # reward discount factor (gamma), 1.0 = no discount
    optimizer = optim.Adam(policy.parameters(), lr=0.000001) #update parameters
    optimizer_critic = optim.Adam(critic.parameters(),lr = 0.00001) #update parameters
    loss_lsq = torch.nn.MSELoss()
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
        value_observations,indices_of_last_state, value_observations_discounted = [] , [] , []

        # build batch
        while True:
            # generate rollout by iteratively evaluating the current policy on the environment
            with torch.no_grad():

                s_tensor = torch.from_numpy(normalize(s)).float().permute(2,0,1).view(1,4,64,64)
                a_prob = policy(s_tensor)  #calls forward function
                estimated_value = critic(s_tensor)
            
            #print(a_prob.numpy())

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
                indices_of_last_state.append((len(rewards_of_episode)))
                # isn't this just making the final rewards of an episode go small, this should be for each time step
                discounted_rewards.append(discount_and_normalize_rewards(rewards_of_episode, discount_factor))
                #value_observations_discounted.append(discount_and_normalize_rewards(value_observations,discount_factor))
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

        observed_values_critic = np.stack(np.array(value_observations))
        observed_values_critic = np.concatenate(value_observations)
        target_values_critic , len_episodes_batch = critic_target_values(rewards_of_batch)
        #print("PRE \n",rewards_of_batch)
        #print("POST \n", rewards_of_batch)
        #print("TARGET VALUES CRITIC \n", target_values_critic)
        target_values_critic = np.concatenate(target_values_critic)
        target_values_critic = make_list_sigmoid(target_values_critic) #make sigmoid 
        

        rewards_of_batch = np.concatenate(rewards_of_batch)
        returns = np.concatenate(discounted_rewards)

        #update critic
        target = torch.Tensor(target_values_critic).view(-1)
        observations = torch.Tensor(observed_values_critic).view(-1) 
        target = Variable(target,requires_grad = True)
        observations = Variable(observations,requires_grad = True)
        loss_critic = loss_lsq(observations, target)  
        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()


        """
        V of st+1, this value will be wrong for the last state of an episode as it's going to be the total reward obtained in the next episode 
        To fix this we say that the V(st+1) for the last state of each episode is 0 
        Rudimentary solution. Consider other ways of doing this
        """
        observed_values_critic_shifted = np.roll(observed_values_critic,-1)
        temp = 1
        c = 0

        for i in indices_of_last_state:
            c = c + i - temp
            observed_values_critic_shifted[c] = 0
            temp = 0

        #IMPROVE THIS
        discount_factor_critic = 0.9
        
        # policy gradient update
        actual_batch_size = states.shape[0] #can be different from batch size as episode length is not constant
        optimizer.zero_grad()
        states_batch = torch.from_numpy(normalize(states)).float().permute(0,3,2,1)
        a_probs = policy(states_batch)
        #print("A PROBS\n",(a_probs.numpy())[:5])
        
        observed_values_critic_shifted = observed_values_critic_shifted.reshape(-1)
        observed_values_critic = observed_values_critic.reshape(-1)
        advantages = rewards_of_batch + (discount_factor_critic*observed_values_critic_shifted) - observed_values_critic  #discount factot
        advantages_tensor = torch.Tensor(advantages).view(-1)

        likelihood = NLL(a_probs, torch.LongTensor(actions)) #WHY?
        loss = torch.sum(advantages_tensor*likelihood)  #torch.mean vs torch.sum
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # bookkeeping
        training_rewards.append(np.mean(rewards_of_batch))

        print("==========================================")
        print("Epoch: ", epoch, "/", num_epochs)
        print("-----------")
        print("Number of training episodes: {}".format(num_episode))
        print("Average Lenght of Episode: {}".format(np.mean(len_episodes_batch)))
        print(len_episodes_batch)
        print("Total reward: {0:.2f}".format(np.sum(rewards_of_batch)))
        print("Mean Reward of that batch {0:.2f}".format(np.mean(rewards_of_batch)))
        print("Training Loss for Actor: {0:.2f}".format(loss.item()))
        print("Training Loss for Critic: {0:.2f}".format(loss_critic.item()))
        #print("Length of last episode: {0:.2f}".format(rewards_of_batch.shape[0]))

        
        if (epoch+1) % 50 == 0:
            #display_episode(np.array(states_human_size))
            format_frames = np.array(states_human_size)
            imageio.mimwrite('videos/a2c_videos_scenario2_with_changes/training_a2c'+str(epoch+1)+'.mp4', format_frames[:,:,:,0], fps = 15)
            #plt.figure(figsize = (10,10))
            #x = [i for i in range(0,50)]
            #plt.plot(training_loss)
            #plt.show()


    print('done')


def make_list_sigmoid(lista):

    for i in range(len(lista)):
        lista[i] = (1/(1+np.exp(-lista[i])))
    return lista

def deep_copy(x):
    return [y[:] for y in x]

def critic_target_values(batch_rewards):
    """
    At each state, the target value is the sum of the rewards that were still obtained in the episode
    Returns: 
        - List of Lists containing the target values of the critc
        - List containing lenght of each episode in the batch
    """

    batch_rewards_copy = [x[:] for x in batch_rewards] 
    episode_lenghts = []

    i = 0
    for episode in batch_rewards_copy:
        episode_lenghts.append(len(episode)) # keep track of how long untill agent dies
        for episode_reward in (range(len(episode))):
            episode[episode_reward] = sum(episode[episode_reward:])
        i += 1

    return batch_rewards_copy, episode_lenghts  


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


#def save_model():

def normalise_list (l,max_intensity):

    max= np.max(l)

    l = (l/max_value) * max_intensity

    return l


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
