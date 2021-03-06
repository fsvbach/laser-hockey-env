from feedforward import Feedforward
import torch
import numpy as np

class Actor(Feedforward):

    def __init__(self, observation_dim, action_dim, hidden_sizes=[200,100],
                 learning_rate=0.0001):
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes,
                         output_size=action_dim, actor=True)
        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        

    def fit(self, q_policy):
        #self.train() # put model in training mode
        self.optimizer.zero_grad()
        loss = -q_policy.mean()
        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_action(self, obs):
        return torch.clamp(self.forward(torch.from_numpy(obs).float()), min=-2.0, max=2.0)
