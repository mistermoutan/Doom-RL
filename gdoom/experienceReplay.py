import numpy as np
from collections import deque, namedtuple
from random import sample as randomSample

from gdoom_env import *
from frames import *
from serialization import *


#########################################################################################################
# My version
class ExperienceReplay():
    '''
    Format for saved states
    [state, action, reward, nextState, done]
    '''
    def __init__(self, maxSize):
        self.buffer = deque(maxlen=maxSize)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batchSize):
        bufferSize = len(self.buffer)
        index = np.random.choice(np.arange(bufferSize),
                                size = batchSize,
                                replace = False)
        
        return [self.buffer[i] for i in index]


def preTrainExperienceReplay(er, env, preTrainLength=1000):

    state = env.reset()
    state = preprocessState(state)

    for i in range(preTrainLength):
        action = env.action_space.sample()

        newState, reward, done, info = env.step(int(action))

        if done:

            er.add([state, action, reward, newState, done])

            state = env.reset()
            state = preprocessState(state)

        else:
            newState = preprocessState(newState)

            er.add([state, action, reward, newState, done])

            state = newState

    return er

#########################################################################################################
# From Pytorch Tutorial
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

    def sample(self, batch_size, minibatch_size):
        '''
        Returns a list of minibatches
        Randomly samples as many sample as the batch size
        Than randomly allocates them in the mini-batches
        '''
        return [randomSample(self.memory, batch_size)[i:i+minibatch_size] for i in range(0,batch_size,minibatch_size)]

    def __len__(self):
        return len(self.memory)

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