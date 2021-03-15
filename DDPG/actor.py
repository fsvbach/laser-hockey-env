#!/usr/bin/env python3
'''
@author maitim
'''

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

"""
    Neural Network Actor class
"""
class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 1e-4):
        super(Actor, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, output_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, state):
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        return torch.tanh(self.lin3(x))
        # uncomment for pendulum (scaling actions to [-2, 2] gives better results)
        # return 2.0 * torch.tanh(self.lin3(x))
