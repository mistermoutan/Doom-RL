import gym
from gdoom.gdoom_env import CPU, HUMAN
__all__ = ['SetPlayingMode']
import numpy as np

# Library used to modify frames (former times we used matplotlib)
import cv2

# setUseOpenCL = False means that we will not use GPU (disable OpenCL acceleration)
cv2.ocl.setUseOpenCL(False)

# From sonic env.
class GPreprocessFrame(gym.ObservationWrapper):
    """
    Here we do the preprocessing part:
    - Set frame to gray
    - Resize the frame to 96x96x1
    """
    def __init__(self, env,size=96):
        gym.ObservationWrapper.__init__(self, env)
        self.width = size
        self.height = size
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        # Set frame to gray
        # print(frame.shape, frame.max())
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize the frame to 96x96x1
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None].copy()

        return frame


class GActionsDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(GActionsDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []

        """
        What we do in this loop:
        For each action in actions
            - Create an array of 12 False (12 = nb of buttons)
            For each button in action: (for instance ['LEFT']) we need to make that left button index = True
                - Then the button index = LEFT = True

            In fact at the end we will have an array where each array is an action and each elements True of this array
            are the buttons clicked.
        """
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


class GRewardScaler(gym.RewardWrapper):
    def __init__(self, env, scale=1.0):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * self.scale


def SetPlayingMode(target_mode):
    """ target mode can be 'algo' or 'human' """

    class SetPlayingModeWrapper(gym.Wrapper):
        """
            Doom wrapper to change playing mode 'human' or 'algo'
        """
        def __init__(self, env):
            super(SetPlayingModeWrapper, self).__init__(env)
            if target_mode not in [CPU, HUMAN]:
                raise gym.error.Error('Error - The mode "{}" is not supported. Supported options are "algo" or "human"'.format(target_mode))
            self.unwrapped.mode = target_mode

    return SetPlayingModeWrapper
