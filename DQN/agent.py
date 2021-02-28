from .feedforward import Feedforward
from .memory import Memory
import numpy as np
import torch

""" Q Network, input: observations, output: q-values for all actions """
class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, 
                 hidden_sizes=[100,100], learning_rate = 0.0002):
        super().__init__(input_size=observation_dim, 
                         hidden_sizes=hidden_sizes, 
                         output_size=action_dim)
        self.optimizer=torch.optim.Adam(self.parameters(), 
                                        lr=learning_rate, 
                                        eps=0.000001)
        # The L1 loss is often easier for choosing learning rates etc than for L2 (MSELoss)
        #  Imagine larger q-values (in the hundreds) then an squared error can quickly be 10000!, 
        #  whereas the L1 (absolute) error is simply in the order of 100. 
        self.loss = torch.nn.SmoothL1Loss()
        #self.loss = torch.nn.MSELoss()
        
    def fit(self, observations, actions, targets):
        
        # put model in training mode
        self.train()
        #set gradients to zero
        self.optimizer.zero_grad()
        
        # Forward pass
        pred    = self.Q_value(observations, actions)
        targets = torch.from_numpy(targets).float()
        loss = self.loss(pred, targets)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def Q_value(self, observations, actions):
        # compute the Q value for the give actions
        #print('Actions shape: ', actions.shape, '\n')
        actions  = torch.tensor(actions).reshape((actions.shape[0],1))
        q_values = self.forward(torch.from_numpy(observations).float())
        result =  torch.gather(q_values, 1, actions).flatten()
        
        return result
    
    def maxQ(self, observations):
        return np.max(self.predict(observations), axis=-1)
    
    def greedyAction(self, observations):
        return np.argmax(self.predict(observations), axis=-1)
    
    
class DQNAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.    
    """
    def __init__(self, observation_space, action_space, **userconfig):
        
        self._observation_space = observation_space
        self._observation_n = len(observation_space.low)
        self._action_space = action_space
        self._action_n = action_space.n
        self._config = {
            "eps": 0.05,            # Epsilon in epsilon greedy policies                        
            "discount": 0.95,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "learning_rate": 0.0002, 
            "update_rule": 3,
            # add additional parameters here        
        }
        self._config.update(userconfig)        
        self._eps = self._config['eps']
        self.buffer = Memory(max_size=self._config["buffer_size"])
        
        # complete here
        self.train_iter = 0
        self.Q = QFunction(self._observation_n ,action_space.n)  
        self.T = QFunction(self._observation_n , action_space.n)
            
    def _update_target_net(self):        
        self.T.load_state_dict(self.Q.state_dict())
    
    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        # epsilon greedy
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else: 
            action = self._action_space.sample()        
        return action
    
    def store_transition(self, transition):
        self.buffer.add_transition(transition)
            
    def train(self, iter_fit=32):
        losses = []
        
        # update the target networks parameters every k train calls
        k=self._config["update_rule"]
        self.train_iter +=1
        if self.train_iter % k == 0:
            self._update_target_net()
        
     
        for i in range(iter_fit):
            
            # sample from the replay buffer
            data   = self.buffer.sample(batch=self._config["batch_size"])
            states      = np.stack(data[:,0]) # s_t
            next_states     = np.stack(data[:,3]) # s_t+1
            actions      = np.stack(data[:,1]) # a
            rewards    = np.stack(data[:,2])[:,None].flatten() # rew
            
            # target network estimates the values of the next states
            next_state_values = self.T.maxQ(next_states)
            
            # TD target is computed based on target network predictions
            target = rewards + self._config['discount']*next_state_values

            # only optimize the parameters of the Q network
            fit_loss = self.Q.fit(states, actions, target)
            
            losses.append(fit_loss)    
  

        return losses
    
    
    
    
    
    
    
    
    