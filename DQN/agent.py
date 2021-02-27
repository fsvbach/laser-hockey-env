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
        # TODO: complete this
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass
        
        pred    = self.Q_value(observations, actions)
        targets = torch.from_numpy(targets).float()
        
        # Compute Loss
        #print('pred, targets', pred.shape, targets.shape)
        loss = self.loss(pred, targets)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def Q_value(self, observations, actions):
        # compute the Q value for the give actions
        # Hint: use the torch.gather function select the right outputs 
        # Complete this
        
        #print('a before',actions)
        
        actions  = torch.tensor(actions).reshape((observations.shape[0],1))
        
        #print('a after',actions)
        
        q_values = self.forward(torch.from_numpy(observations).float())
        
        #print('q before', q_values)
        
        #q_values = torch.tensor(q_values, requires_grad=True)
        
        #print('q after',q_values)
        
        result =  torch.gather(q_values, 1, actions).flatten()
        
        #print('result',result)
        
        return result
    
    def maxQ(self, observations):
        # compute the maximal Q-value
        # Complete this
        #observations = torch.from_numpy(observations).float()
        return np.max(self.predict(observations), axis=-1)
    
    def greedyAction(self, observations):
        # this computes the greedy action
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
        # complete here
        # Hint: use load_state_dict() and state_dict() functions
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
        # complete this! 
        # Hint: look at last exercise's solution
        # Hint: while developing print the shape of !all! tensors/arrays to make sure 
        #  they have the right shape: (batchsize, X)  
        
        # Hint: for the target network, update its parameters at the beginning of this function 
        # every k  train calls. 
        
        k=self._config["update_rule"]
        self.train_iter +=1
        if self.train_iter % k == 0:
            self._update_target_net()
        
        # Hint:
        for i in range(iter_fit):
            
            # sample from the replay buffer
            data   = self.buffer.sample(batch=self._config["batch_size"])
            s      = np.stack(data[:,0]) # s_t
            sp     = np.stack(data[:,3]) # s_t+1
            a      = np.stack(data[:,1]) # a
            rew    = np.stack(data[:,2])[:,None].flatten() # rew
            
            #print(rew)
            values  = self.Q.maxQ(sp)
            valuesp = self.T.maxQ(sp)
            
            # target
            target = rew+self._config['discount']*valuesp

            # optimize the lsq objective
            #print('loss',s.shape, target.shape )
            #print('target', valuesp.shape, rew.shape)
            #print(self._config['batch_size'], self._config['discount'])
            
            fit_loss = self.Q.fit(s, a, target)
            
            losses.append(fit_loss)    
            #print(fit_loss)
            #input()

        return losses