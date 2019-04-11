import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepQNet(nn.Module):
    '''
    Neural Net that approximates the Q value
    Input: 4 Frames stacked
    Output: Q-value for every possible actions
    '''

    def __init__(self, nbrActions=3):
        super(DeepQNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ELU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ELU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ELU())
        self.fc1 = nn.Sequential(
            nn.Linear(8*16*64, 512),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(512, nbrActions),
            nn.ReLU())
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x