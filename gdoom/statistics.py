#import train
#import a2c
import time
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pickle

"""
Class to benchmark performance of methods

HOW TO USE:

    1. Create instance of class once training is initiated, see the constructor for the arguments you need to pass
    2. During training append the values to the relevant lists of the constructor or find proper ways of completing them. The only array for now that is dependant of the level is the kills_per_episode. So see if the agent is shooting stuff in your level
    3. At the end of training call the get_statistics method (stat.get_statistics()) and BOOM done just like it was magic

FEEL FREE TO ADD STATISTICS YOU FIND RELEVANT AND TWEAK THIS
"""



class Statistics:

    def __init__(self,scenario:str ,method: str,epochs: int, directory:str = ""):
        """
        scenario: give it the string of what you feed to the enviromnent (eg: "doom_scenario7_640-v0", alternatively you can give it the common name of the level like "defend_the_center", check the function convert_scenario below to see the names)
        epochs: number of epochs
        method: str (eg: a2c, ppo, deep_q_learning)
        directory: directory where you want to save the stats
        """

        self.start_time = time.time()
        self.directory = directory
        self.scenario = scenario
        self.method = method # the algorithm used
        self.epochs = epochs

        # UPDATE THIS IF RELEVANT
        self.batch_size = None
        self.mini_batch_size = None

        # UPDATE THIS DURING TRAINING
        
        #PER EPISODE
        self.rewards_per_episode = [] #final value of reward obtained at end of episode
        self.lenght_episodes = []
        self.kills_per_episode = [] #if relevant to level

        # PER EPOCH
        self.loss_actor = []
        self.loss_critic = []
        

    def get_statistics(self):
        
        self.end_time = time.time() #for tracking training time

        self.build_stats_dictionary() 
        self.save_graphs()  
        self.save_log_book()
        self.save_arrays()


    def build_stats_dictionary(self, save = True):
        """ builds dictionary of statistics (self.stats), saves it as a pickle
            açso builds dict containing info for the last 100 episodes """

        self.stats = {}
        self.stats_last100episodes = {}
        assert len(self.rewards_per_episode) == len(self.lenght_episodes)

        self.stats["date"] = time.strftime("%c")
        self.stats["method"] = self.method
        self.stats["scenario"] = self.convert_scenario(self.scenario)
        self.stats["batch_size"] = self.batch_size
        self.stats["mini_batch_size"] = self.mini_batch_size

        self.stats["steps"] = sum(self.lenght_episodes) #actions taken or sets of 4 frames fed to network
        self.stats["avg_len_episode"] = self.stats["steps"] / len(self.lenght_episodes)
        self.stats["avg_reward_episode"] = sum(self.rewards_per_episode) / len(self.rewards_per_episode)
        self.stats["training_time"] = str(datetime.timedelta(seconds=self.end_time - self.start_time))

        if self.kills_per_episode:
            self.stats["avg_kills_episode"] = sum(self.kills_per_episode) / self.stats["steps"]
            self.stats_last100episodes["avg_kills_episode"] = sum(self.kills_per_episode[-100:]) / 100



        self.stats_last100episodes["avg_len_episode"] = sum(self.lenght_episodes[-100:]) / 100
        self.stats_last100episodes["avg_reward_episode"] = sum(self.rewards_per_episode[-100:]) / 100
    
        if save:
            self.write_pickle(self.stats,self.directory,"stats.pickle")
            self.write_pickle(self.stats_last100episodes,self.directory,"stats_last_100.pickle")


    def save_arrays(self):
        """ saves arrays in dict, may be relevant for future comparion between methods"""

        dict_of_arrays = {}
        dict_of_arrays["rewards_per_episode"] = self.rewards_per_episode 
        dict_of_arrays["len_episodes"] = self.lenght_episodes
        if self.kills_per_episode:
            dict_of_arrays["kills_per_episode"] = self.kills_per_episode
        
        dict_of_arrays["loss_actor"] = self.loss_actor
        dict_of_arrays["loss_critic"] = self.loss_critic

        
        self.write_pickle(dict_of_arrays,self.directory,"arrays.pickle")

    def save_graphs (self):

        assert len(self.loss_actor) == len(self.loss_critic)
        #evolution of network losses
        figure = plt.figure()
        x_axis = np.arange(len(self.loss_actor))
        plt.plot(x_axis,self.loss_actor)
        plt.title("ACTOR/POLICY LOSS")
        figure.savefig(self.directory + "loss_actor_evolution.png")

        if self.loss_critic:
            figure = plt.figure()
            x_axis = np.arange(len(self.loss_critic))
            plt.plot(x_axis,self.loss_critic)
            plt.title("CRITIC LOSS")
            figure.savefig(self.directory + "loss_critic.png")
            
        #evolution of rewards per episode
        figure = plt.figure()
        x_axis = np.arange(len(self.rewards_per_episode))
        plt.plot(x_axis,self.rewards_per_episode)
        plt.title("REWARDS PER EPISODE")
        figure.savefig(self.directory + "rewards_per_episode_evolution.png")

        #evolution of episode lenght
        figure = plt.figure()
        plt.title("LENGHT OF EPISODES")
        x_axis = np.arange(len(self.lenght_episodes))
        plt.plot(x_axis,self.lenght_episodes)
        figure.savefig(self.directory + "lenght_of_episode_evolution.png")

        if self.kills_per_episode:
            figure = plt.figure()
            plt.title("KILLS PER EPISODE")
            x_axis = np.arange(len(self.kills_per_episode))
            plt.plot(x_axis,self.kills_per_episode)
            figure.savefig(self.directory + "kills_per_episode_evolution.png")

    def save_log_book (self):
        with open(self.directory + "log.txt",'w') as logbook:
            for key,value in self.stats.items():
                logbook.write("{0} : {1}  \n " .format(key,value))
            logbook.write("\n LAST 100: \n")
            for key,value in self.stats_last100episodes.items():
                logbook.write("{0} : {1}  \n " .format(key,value))





    def convert_scenario(self,scenario):

        if "scenario1" in self.scenario:
            converted_scenario = "deadly_corridor"
        elif "scenario2" in self.scenario:
            converted_scenario = "defend_the_center"
        elif "scenario3" in self.scenario:
            converted_scenario = "defend_the_line"
        elif "scenario4" in self.scenario:
            converted_scenario = "health_gathering"
        elif "scenario5" in self.scenario:
            converted_scenario = "my_way_home"
        elif "scenario6" in self.scenario:
            converted_scenario = "predict_position_rocket_launcher"
        elif "scenario7" in self.scenario:
            converted_scenario = "avoid_fireballs"

        return converted_scenario

        #scenario2: defend_the_center
        #scenario3: defend_the_line
        #scenario4: healthy_gathering
        #scenario5: my_way_home 
        #scenario6: predict_position
        #scenario7: avoid_fireballs 

    def write_pickle (self, f, path, fname):

        with open(path + fname, 'wb') as handle:
            pickle.dump(f, handle, protocol = pickle.HIGHEST_PROTOCOL)



        








    

        