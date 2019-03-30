import numpy as np
import copy

def discount_and_normalize_rewards(episode_rewards, discount_factor):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * discount_factor + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    # print(discounted_episode_rewards)
    #discounted_episode_rewards = 1e5*(discounted_episode_rewards - mean)# / (std)
    # print(discounted_episode_rewards)
    return discounted_episode_rewards

#print(discount_and_normalize_rewards([1,2,3,4,5,6],0.95))

def critic_target_values_old(batch_rewards):

    # at each state, the target value is the sum of the rewards that were still obtained in the episode
    print(batch_rewards)
    batch_rewards_copy = copy.deepcopy(batch_rewards)  #deep copy
    episode_lenghts = []

    i = 0
    for episode in batch_rewards_copy:
        episode_lenghts.append(len(episode)) # keep track of how long untill agent dies
        for episode_reward in (range(len(episode))):
            episode[episode_reward] = sum(episode[episode_reward:])
        i += 1

 
    return batch_rewards_copy      

#rewards_of_episode = [[2,3,4,5],[2,1,1,5]]
#print(critic_target_values_old(rewards_of_episode))


def critic_target_values(batch_rewards,gamma = 0.99):
    """
    At each state, the target value is the sum of the rewards that were still obtained in the episode
    Returns: 
        - List of Lists containing the target values of the critc
        - List containing lenght of each episode in the batch
    """

    batch_rewards_copy = [x[:] for x in batch_rewards] 
    episode_lenghts = []

    
    for episode in batch_rewards_copy:
        episode_lenghts.append(len(episode)) # keep track of how long untill agent dies
        for episode_reward in reversed(range(len(episode))):
            try:
                episode[episode_reward] += (gamma * episode[episode_reward +1]) 
            except IndexError:
                episode[episode_reward -1]  += (gamma * episode[episode_reward]) 
                episode[episode_reward] = 0
        

    return batch_rewards_copy


#rewards_of_episode = [[2,3,4,5],[2,1,1,5]]
#print(critic_target_values(rewards_of_episode))



def compute_returns(next_value, rewards, masks, gamma=0.99):
    
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns