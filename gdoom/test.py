import numpy as np

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

def critic_target_values(batch_rewards):
    # at each state, the target value is the sum of the rewards that were still obtained in the episode
    for episode in batch_rewards:
        for episode_reward in (range(len(episode))):

            episode[episode_reward] = sum(episode[episode_reward:])

    return batch_rewards
"""         
discount_critic = critic_target_values([[1,2,3],[5,4,2]])[0]
print(discount_critic)
discount_critic = discount_and_normalize_rewards(discount_critic,0.95)
print(discount_critic)
#print(critic_target_values([[1,2,3],[2,2,2]]))
a = (np.stack([0,1,2,3,4,5,6,7]))
print(np.roll(a,1))
"""
print(discount_and_normalize_rewards([2,4,5,6,7,8,9],0.95))