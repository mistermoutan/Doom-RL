import numpy as np
from collections import namedtuple
from random import sample as randomSample

from gdoom_env import *
from serialization import *
from utils import *

#########################################################################################################
# From Pytorch Tutorial
# Named Tuple: https://docs.python.org/2/library/collections.html#collections.namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity


    def sample(self, minibatch_size):
        '''
        Returns a list of minibatches
        Randomly samples as many transitions as the minibatchsize
        '''
        return randomSample(self.memory, minibatch_size)

    def getInMemorySize(self):
        return 0.1*self.capacity

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def getSaveName(memorySize):
        return 'memory{0}'.format(memorySize)


#########################################################################################################

if __name__ == '__main__':

    # Make a CPU environemnt the good ol' way (not recommended, see __init__.py).
    genv = WGDoomEnv(level=1, frame_size=640)
    genv.reset()

    # Also make a GPU environment, but using openai:
    env = gym.make("doom_scenario4_640-v0")
    frame = env.reset()
    print("Frame size for cpu player: ", np.asarray(frame).shape )

    # Create Instance.
    trainLength = 10000
    er = ExperienceReplay(trainLength)
    er = preTrainExperienceReplay(er, env, preTrainLength=trainLength)
    print(er.buffer.pop())

    # Save it.
    serializeObject(er, HOME_DIR + '/saves/experienceReplay10000.pickle')
