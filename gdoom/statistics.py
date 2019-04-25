#import train
#import a2c
import time
from matplotlib import pyplot as plt

class Statistics:

    def __init__(self,scenario: str = "",method: str = "",epochs: int = 0):

        self.epochs = epochs
        self.scenario = scenario
        self.method = method
        #containers
        self.rewards = []
        self.average_rewards_per_episode = []
        self.average_reward_epoch = []
        self.loss_actor = []
        self.loss_critic = []
        self.action_taken = []
        self.actions_prob = []
        self.average_lenght_episode_per_epoch = []
        self.lenght_episode = []
        self.kills_per_episode = []
        self.time_steps = 0

    def append_reward(self,reward):
        self.rewards.append(reward)
    
    def append_average_reward_per_episode(self,episode_average_reward):
        self.average_rewards_per_episode.append(episode_average_reward)

    def append_action_taken(self,action):
        self.action_taken.append(action)

    def append_actions_prob(self,actions_prob):
        self.actions_prob.append(actions_prob)

    def append_loss_actor(self,loss_actor):
        self.loss_actor.append(loss_actor)

    def append_loss_critic(self,loss_critic):
        self.loss_critic.append(loss_critic)

    def append_average_lenght_episode_per_epoch(self,average_lenght_episode_epoch):
        self.average_lenght_episode_per_epoch.append(average_lenght_episode_epoch)
    
    def append_lenght_episode(self,lenght_episode):
        self.lenght_episode.append(lenght_episode)

    def append_kills_per_episode(self,kills):
        self.kills_per_episode.append(kills)


    def save_log_book(self):
        return None
        #day
        #scenario
        #method


    def save_plots(self):
        self.number_episodes = len(self.rewards)

        








    

        