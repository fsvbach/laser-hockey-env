#!/usr/bin/env python3
'''
@author maitim
'''

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

"""
    Neural network critic class
"""
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=1e-3):
        super(Critic, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, output_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = torch.nn.MSELoss()

    def forward(self, obs, action):
        x = torch.cat([obs, action], 1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)
    
    def fit(self, q, td_target):
        self.optimizer.zero_grad()
        loss = self.loss(q, td_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
