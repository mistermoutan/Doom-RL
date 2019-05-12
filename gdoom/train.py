from experienceReplay import *
from prioritizedExperienceReplay import *
from net import *
from utils import *

## NOTES

# Articles
# https://gist.github.com/simoninithomas/7611db5d8a6f3edde269e18b97fa4d0c#file-deep-q-learning-with-doom-ipynb
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20%28%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay%29.ipynb
# https://towardsdatascience.com/tutorial-double-deep-q-learning-with-dueling-network-architectures-4c1b3fb7f756

# Hyperparameters
# https://www.nature.com/articles/nature14236/tables/1 Deep Mind's Table

#########################################################################################################

BATCH_SIZE = 32
MINIBATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.001
EPS_DECAY = 0.00005
TARGET_UPDATE = 10000 # https://github.com/deepmind/dqn/blob/9d9b1d13a2b491d6ebd4d046740c511c662bbe0f/run_gpu#L31
# LEARNING_RATE = 0.000025 for RMSProp, Deep Mind
LEARNING_RATE = 0.000065 #for Adam, Deep Mind
#LEARNING_RATE = 1e-4
OPTIMIZE_FREQUENCY = 1
PLOT_FREQUENCY = 250
DISPLAY = False
DISPLAY_FREQUENCY = 10

SOFTMAX_MULT = 50

#########################################################################################################


class Trainer:

    def __init__(self, env, device, n_actions, memory):
        self.env = env
        self.device = device
        self.n_actions = n_actions
        self.population = [i for i in range(n_actions)]

        self.policy_net = DeepQNet(nbrActions=n_actions).to(device)
        self.policy_net.apply(weights_init)
        self.target_net = DeepQNet(nbrActions=n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.apply(weights_init)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = memory

        self.steps_done = 0
        self.episode_durations = []
        self.life_rewards = []
        self.life_reward = 0
        self.losses = []


    def getMemory(self):
        return self.memory

    def setMemory(self, memory):
        self.memory = memory
        return

    def getPolicyNet(self):
        return self.policy_net

    def setPolicyNet(self, policy_net):
        self.policy_net = policy_net
        return

    def getTargetNet(self):
        return self.target_net

    def setTargetNet(self, target_net):
        self.target_net = target_net
        return

    def select_action(self, state):
        '''
        Update according to https://openai.com/blog/openai-baselines-dqn/ ?
        Have two slopes for decrease in exploration probability.
        '''
        sample = random()
        eps_threshold = EPS_END + ((EPS_START - EPS_END)*exp(-1. * self.steps_done * EPS_DECAY))
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # Sample from the probability distribution output of the policy net.
                #sample_prob = random()
                #eps_threshold = EPS_END + ((EPS_START - EPS_END)*exp(-1. * self.steps_done / EPS_DECAY))

                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                # distribution = softmax(SOFTMAX_MULT * self.policy_net(state).view(-1).numpy())
                # return torch.tensor([choices(self.population, distribution)], device=self.device, dtype=torch.long)

                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        '''
        OPTIMIZATION of the policy network.
        '''
        if len(self.memory) < MINIBATCH_SIZE:
            return

        treeIndices, minibatches, IS_weights = self.memory.sample(MINIBATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*minibatches))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros((MINIBATCH_SIZE,3), device=self.device)
        # NOTE WHY NOT CONSIDER THE TERMINATING STATES????
        # Deep Mind paper says that the state action value should be the reward in the case of terminating sequence.
        # Here, It comes from the mask.

        ## To suppress.
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        ## TODO
        # Check for amax(s';theta) = argmax_a' Q(s'; a'; theta) using policy net.
        next_state_a_max = torch.zeros(MINIBATCH_SIZE, device=self.device, dtype=torch.long)
        next_state_a_max[non_final_mask] = self.policy_net(non_final_next_states).max(1)[1].detach()

        ## TODO
        # Generate new states with the amax
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).detach()

        qOperator = nn.NLLLoss(reduction='none')
        new_next_state_values = (-1*qOperator(next_state_values, next_state_a_max))

         # And then use r + Q(s'; amax (s'; theta); theta-); using target net to get q value
         # Compute the expected Q values
        expected_state_action_values = (new_next_state_values * GAMMA) + reward_batch


        # Compute Huber loss, with importance sampling.
        loss = torch.mean(torch.from_numpy(IS_weights).to(self.device)*F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none'))
        # For priority update.
        absoluteErrors = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none').detach().numpy()


        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update priority
        self.memory.batch_update(treeIndices, absoluteErrors)

        return loss.item()
        


    def train(self, num_episodes=1000, statisticsInstance=None):
        '''
        TRAINING loop.
        '''
        for i_episode in range(num_episodes):
            print('Episode: {0}'.format(i_episode + 1))

            # Initialize the environment and state
            state  = self.env.reset()
            states_human_size = [np.asarray(state)] # Allow visualization of episode.
            state = torchify(preprocessState(state), self.device)
            avgLoss = 0
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action.item())
                states_human_size.append(np.asarray(next_state))
                self.life_reward += reward
                reward = torch.tensor([reward], device=self.device)

                # Observe new state
                if not done:
                    next_state = torchify(preprocessState(next_state), self.device)
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform optimization (on the target network)
                if self.steps_done % OPTIMIZE_FREQUENCY == 0:
                    loss = self.optimize_model()
                    avgLoss += loss
                    self.losses.append(loss)

                # Update the target network, copying all weights and biases in DQN
                if self.steps_done % TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if done:
                    self.life_rewards.append(self.life_reward)
                    self.life_reward = 0
                    self.episode_durations.append(t + 1)
                    print(info)
                    print('Avg loss:{0}'.format(avgLoss/(t+1)))

                    # Stats feeding
                    if statisticsInstance != None:
                        # {'accumulated_reward': 179.60000000000008, 'time_alive': 251, 'kills': 17.0}
                        statisticsInstance.rewards_per_episode.append(info['accumulated_reward'])
                        statisticsInstance.length_episodes.append(info['time_alive'])
                        statisticsInstance.kills_per_episode.append(info['kills'])
                        statisticsInstance.loss_actor.append(avgLoss/(t+1))
                    break

            if (i_episode+1) % PLOT_FREQUENCY == 0:
                plotRewardsLosses(i_episode+1, self.life_rewards, self.losses, pictureName='RewardsLosses_Episode{0}'.format(i_episode+1))
                pass

            if ((i_episode+1) % DISPLAY_FREQUENCY == 0) and (DISPLAY):
                # display_episode(np.array(states_human_size))
                pass


    def preTrainMemory(self, pre_train=1000):
        '''
        PRE-FILL memory to start.
        '''
        state  = self.env.reset()
        state = torchify(preprocessState(state), self.device)

        for i in range(pre_train):
            if (i % (pre_train/10) == 0):
                print('-- Pre Training, {0}/{1} --'.format(i+1,pre_train))
            action = torch.tensor([[randrange(self.n_actions)]], device=self.device, dtype=torch.long)

            next_state, reward, done, info = self.env.step(action.item())
            reward = torch.tensor([reward], device=self.device)

            # Observe new state
            if not done:
                next_state = torchify(preprocessState(next_state), self.device)
                self.memory.push(state, action, next_state, reward)
                state = next_state
            else:
                next_state = None
                self.memory.push(state, action, next_state, reward)
                state = self.env.reset()
                state = torchify(preprocessState(state), self.device)
        return
