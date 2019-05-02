import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocessState(state):
    '''
    Preprocess state (=stack of 4 frames)
    Turns frames into 64*64 frames
    '''
    state = np.asarray(state)
    state = cv2.resize(state, (64,64))
    return (state)/255.0

def display_episode(frames):
    img = None
    for frame in frames[:,:,:,0]:
        if img==None:
            img = plt.imshow(np.asarray(frame), cmap='gray')
        else:
            img.set_data(np.asarray(frame))
        plt.pause(0.1)
        plt.draw()
