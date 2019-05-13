import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

HOME_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).replace('\\','/').split('/')[:-1])

INPUT_LOG_FILENAME = HOME_DIR + '/misc/arrays.pickle'

def readPickled(fileName):
    pickledData = {}
    try:
        if os.path.getsize(fileName) > 0:      
            with open(fileName, "rb") as f:
                unpickler = pickle.Unpickler(f)
                # if file is not empty scores will be equal
                # to the value unpickled
                pickledData = unpickler.load()
    except FileNotFoundError:
        raise FileNotFoundError('FILE NOT FOUND!: {0}'.format(fileName))
        pass
    return pickledData

# PARAM
windowSize = 200
logTypes = ['rewards_per_episode', 'kills_per_episode', 'len_episodes', 'loss_actor', 'loss_critic']
blackListLogType = ['loss_actor', 'loss_critic']

arrays = readPickled(INPUT_LOG_FILENAME)

for index, logType in enumerate(logTypes):

    # To pass some types
    if logType in blackListLogType:
        continue
    
    log = np.array(arrays[logType])
    mvgAvg = np.empty((1,log.shape[0]))

    for i in range(windowSize, log.shape[0]):
        mvgAvg[0,i] = np.mean(log[i-windowSize:i])

    plt.figure(index+1, figsize=(10,5))
    plt.plot([i for i in range(log.shape[0])], log)
    plt.plot([i for i in range(log.shape[0])], mvgAvg[0])
    plt.xlabel('Episodes')
    plt.ylabel('?')
    plt.legend(['Complete data', 'Moving average on {0} values'.format(windowSize)])
plt.show()




