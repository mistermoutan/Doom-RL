import torch
import torch.nn as nn
import torch.nn.functional as F



class Net3Actions(nn.Module):
    '''
    Simplest possible network for state evaluation
    Takes as input the average of the 4 frames flattened
    GOAL: get experience with running a network.
    '''

    def __init__(self):
        super(Net3Actions, self).__init__()
    
        # Simple feed forward network
        self.fc1 = nn.Linear(64 * 64 * 4, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        
        return x
