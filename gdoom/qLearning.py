from gym.utils.play import play
from gdoom_env import *

from train import *
from experienceReplay import *
from utils import *


####################
#    Parameters    #
####################
RECOVER_MEMORY = True
MEMORY_SIZE = 60000
NUM_EPISODES = 2000

#########################################################################################################
blockPrint()
# Make a CPU environemnt the good ol' way (not recommended, see __init__.py).
genv = WGDoomEnv(level=1, frame_size=640)

# Also make a GPU environment, but using openai:
enablePrint()
env = gym.make("doom_scenario2_640-v0")
frame = env.reset()

# If gpu is to be used.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get number of actions from gym action space.
n_actions = env.action_space.n

#########################################################################################################

####################
#  Instantiation   #
####################
noFile = None
if RECOVER_MEMORY:
    try:
        memory = readPickled(HOME_DIR + '/saves/{0}.pickle'.format(ReplayMemory.getSaveName(MEMORY_SIZE)))
        trainer = Trainer(env, device, n_actions, memory)
    except FileNotFoundError:
        noFile = True
if (not RECOVER_MEMORY) or noFile:
    memory = ReplayMemory(MEMORY_SIZE)
    trainer = Trainer(env, device, n_actions, memory)
    trainer.preTrainMemory(pre_train=int(MEMORY_SIZE/10))
print('\n---- Done Pre Training ----\n---- ESTIMATION FOR EXPERIENCE REPLAY MEMORY SIZE ----\n{0} MB\n'.format(memory.getInMemorySize()))


################
#     Train    #
################
trainer.train(num_episodes=NUM_EPISODES)


memory = trainer.getMemory()
#serializeObject(memory, HOME_DIR + '/saves/{0}.pickle'.format(ReplayMemory.getSaveName(MEMORY_SIZE)))
policyNet = trainer.getPolicyNet()
serializeObject(policyNet, HOME_DIR + '/saves/policyNet.pickle')
targetNet = trainer.getTargetNet()
serializeObject(targetNet, HOME_DIR + '/saves/targetNet.pickle')
