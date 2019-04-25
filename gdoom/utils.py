import matplotlib.pyplot as plt 
from IPython.display import clear_output
import numpy as np

def random_policy_human():


    ###########################################################
    # Should not work like this, need to give some parameters #
    ###########################################################


    # cv2.imshow("image",np.asarray(frame))
    img = None
    for i in range(500):
        if img==None:
            img = plt.imshow(np.asarray(frame))
        else:
            img.set_data(np.asarray(frame))
        plt.pause(0.01)
        plt.draw()
        # This will just create a sample action in any environment.
        #frame In this environment, the action can be 0 or 1, which is left or right
        action = env_human.action_space.sample()

        # this executes the environment with an action,
        # and returns the observation of the environment,
        # the reward, if the env is over, and other info.
        frame, _, _, _ =env_human.step(action)

def plotRewardsLosses(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(122)
    plt.title('loss')
    plt.plot(losses)
    plt.show()
