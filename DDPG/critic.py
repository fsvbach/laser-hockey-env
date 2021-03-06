from feedforward import Feedforward
import torch
import numpy as np

class Critic(Feedforward):

    def __init__(self, observation_dim, action_dim, hidden_sizes=[200,100],
                 learning_rate=0.001):
        super().__init__(input_size=(observation_dim + action_dim), hidden_sizes=hidden_sizes,
                         output_size=1, actor=False)
        self.optimizer=torch.optim.Adam(self.parameters(),lr=learning_rate, eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss() # MSELoss()

    def fit(self, q, td_target):
        #self.train()
        self.optimizer.zero_grad()
        loss = self.loss(q, td_target.float())
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, obs, a):
        return self.forward(torch.cat((obs, a), dim=1).float())

    