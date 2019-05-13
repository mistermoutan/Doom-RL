from gym.utils.play import play
from gdoom_env import *

from train import *
from experienceReplay import *
from utils import *
from statWriter import *


####################
#    Parameters    #
####################
RECOVER_MEMORY = True
MEMORY_SIZE = 120000
NUM_EPISODES = 5000
SCENARIO = 'scenario2'

#########################################################################################################
blockPrint()
# Make a CPU environemnt the good ol' way (not recommended, see __init__.py).
genv = WGDoomEnv(level=1, frame_size=640)

# Also make a GPU environment, but using openai:
enablePrint()
env = gym.make("doom_{0}_640-v0".format(SCENARIO))
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
        # memory = readPickled(HOME_DIR + '/saves/{0}.pickle'.format(ReplayMemory.getSaveName(MEMORY_SIZE)))
        memory = readPickled(HOME_DIR + '/saves/{0}.pickle'.format(PER.getSaveName(MEMORY_SIZE)))
        trainer = Trainer(env, device, n_actions, memory)
    except FileNotFoundError:
        noFile = True
if (not RECOVER_MEMORY) or noFile:
    # memory = ReplayMemory(MEMORY_SIZE)
    memory = PER(MEMORY_SIZE)
    trainer = Trainer(env, device, n_actions, memory)
    trainer.preTrainMemory(pre_train=MEMORY_SIZE)
print('\n---- Done Pre Training ----\n---- ESTIMATION FOR EXPERIENCE REPLAY MEMORY SIZE ----\n{0} MB\n'.format(memory.getInMemorySize()))


################
#     Train    #
################
stats = Statistics(SCENARIO, 'ddqn_per', NUM_EPISODES, HOME_DIR+'/stats/ddqn_per/'+SCENARIO +'/')
trainer.train(num_episodes=NUM_EPISODES, statisticsInstance=stats)
stats.get_statistics()


memory = trainer.getMemory()
#serializeObject(memory, HOME_DIR + '/saves/{0}.pickle'.format(ReplayMemory.getSaveName(MEMORY_SIZE)))
policyNet = trainer.getPolicyNet()
torch.save(policyNet.state_dict(), HOME_DIR + '/saves/policyNetParams.pickle')
targetNet = trainer.getTargetNet()
torch.save(targetNet.state_dict(), HOME_DIR + '/saves/targetNetParams.pickle')

