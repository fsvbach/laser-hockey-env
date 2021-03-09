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
                                        eps=0.000625)
        self.loss = torch.nn.SmoothL1Loss()
        
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
    def __init__(self, observation_space, action_space, convert_func=lambda x: x, pretrained=False, **userconfig):
        
        self._observation_space = observation_space
        self._observation_n = len(observation_space.low)
        self.convert=convert_func
        self._action_space = action_space
        self._action_n = action_space.n
        self._config = {
            "eps": 1,            # Epsilon in epsilon greedy policies                        
            "discount": 0.5,
            "buffer_size": int(2e4),
            "batch_size": 128,
            "learning_rate": 0.00025, 
            "update_rule": 20,
            "multistep": 3,
            "omega": 0.5,
            # add additional parameters here        
        }
        self._config.update(userconfig)        
        self.transition = []
        self._eps   = self._config["eps"]
        self.buffer = Memory(max_size=self._config["buffer_size"])
        
        # complete here
        self.train_iter = 0
        self.Q = QFunction(self._observation_n ,action_space.n)  
        self.T = QFunction(self._observation_n , action_space.n)
                
        if pretrained:
            try:
                self.load_weights(pretrained)
            except:
                print(f'ERROR: Could not load weights from {pretrained}')
            
    def _update_target_net(self):        
        self.T.load_state_dict(self.Q.state_dict())
        
    def save_weights(self, filepath):
        torch.save(self.Q.state_dict(), filepath)
    
    def load_weights(self, filepath):
        self.Q.eval()
        self.Q.load_state_dict(torch.load(filepath))
        
    def reduce_exploration(self, x): 
        if x < self._eps: 
            self._eps -= x
       

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        if np.random.random() > eps:
            action = self.Q.greedyAction(observation)
        else: 
            action = self._action_space.sample()   
        self.last_action = action
        return self.convert(action)
    
    
    def store_transition(self, transition):
        ob, reward, ob_new, done = transition
        self.transition.append([ob, self.last_action, reward, ob_new, done])
        if len(self.transition) == self._config["multistep"] or done:
            ob, action, reward, ob_new, done = self.transition[0]
            gamma = 1
            if len(self.transition) > 1:
                for transition in self.transition[1:]:
                    _, _, r, ob_new, done = transition
                    gamma *= self._config['discount']
                    reward += gamma * r
            self.buffer.add_transition([ob, action, reward, ob_new, done])
            self.transition = []
            
    def train(self, iter_fit=32):
        if self.buffer.size is not self.buffer.max_size:
            return []
        losses = []
        omega = self._config["omega"]
        k     = self._config["update_rule"]
        gamma = self._config['discount']
        n     = self._config['multistep']
        
        # update the target networks parameters every k train calls
        if self.train_iter % k == 0:
            self._update_target_net()
        self.train_iter +=1
     
        for i in range(iter_fit):
            
            # sample from the replay buffer
            data, indices   = self.buffer.sample(batch=self._config["batch_size"])
            states      = np.stack(data[:,0]) # s_t
            next_states     = np.stack(data[:,3]) # s_t+1
            actions      = np.stack(data[:,1]) # a
            rewards    = np.stack(data[:,2])[:,None].flatten() # rew
            
            # target network estimates the values of the next states
            next_state_values = self.T.maxQ(next_states)
            state_values = self.Q.maxQ(states)
            
            
            # TD target is computed based on target network predictions
            target = rewards + np.power(gamma, n)*next_state_values
            
            # update priorities in buffer
            priorities = np.power(np.abs(state_values-target), omega)
            self.buffer.update_priorities(indices, priorities)
            
            # only optimize the parameters of the Q network
            fit_loss = self.Q.fit(states, actions, target)
            
            losses.append(fit_loss)    

        return losses
    
    
    
    
    
    
    
    
    