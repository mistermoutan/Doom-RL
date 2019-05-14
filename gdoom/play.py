from gym.utils.play import play
from gdoom_env import *

from net import *
from utils import *
from serialization import *

## PARAMETERS
FILENAME = HOME_DIR + '/saves/policyNetParams_s2.pickle'
SCENARIO = 'scenario2'
NBR_EPISODES = 2

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

#net = readPickled(FILENAME).to(device)
net = DeepQNet(nbrActions=n_actions).to(device)
net.load_state_dict(torch.load(FILENAME, map_location='cpu'))
kills = []
for episode in range(NBR_EPISODES):

    state  = env.reset()
    states_human_size = [np.asarray(state)] # Allow visualization of episode.
    state = torchify(preprocessState(state), device)
    for t in count():

        action = net(state).max(1)[1].view(1, 1)
        next_state, reward, done, info = env.step(action.item())
        states_human_size.append(np.asarray(next_state))

        if done:
            display_episode(np.array(states_human_size))
            kills.append(int(info['kills']))
            break
        else:
            state = torchify(preprocessState(next_state), device)

print(kills)
