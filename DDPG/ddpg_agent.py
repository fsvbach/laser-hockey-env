#!/usr/bin/env python3
'''
@author maitim
'''
import torch
import numpy as np
from .memory import Memory
from .actor import Actor
from .critic import Critic


class DDPGAgent(object):
    """
    Agent implementing DDPG.
    """
    def __init__(self, env, pretrained=False, **userconfig):

        self.env = env
        self._observation_space = env.observation_space
        self.action_dim = env.num_actions
        self.obs_dim = self._observation_space.shape[0]

        self._config = {
            "discount": 0.99,
            "hidden_size": 256,
            "buffer_size": int(1e5), 
            "batch_size": 128,
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "update_target_every": 20,
            "update_rate": 1e-2,
        }
        self._config.update(userconfig)

        self.buffer = Memory(max_size=self._config["buffer_size"])
        self.update_rate = self._config["update_rate"]
        self.hidden_size = self._config["hidden_size"]
        self.discount = self._config["discount"]

        self.actor         = Actor(input_size=self.obs_dim, hidden_size=self.hidden_size, output_size=self.action_dim, learning_rate=self._config["actor_lr"])
        self.actor_target  = Actor(input_size=self.obs_dim, hidden_size=self.hidden_size, output_size=self.action_dim)
        self.critic        = Critic(input_size=self.obs_dim + self.action_dim, hidden_size=self.hidden_size, output_size=self.action_dim, learning_rate=self._config["critic_lr"])
        self.critic_target = Critic(input_size=self.obs_dim + self.action_dim, hidden_size=self.hidden_size, output_size=self.action_dim)
        # copy params into target nets for initialization
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # keep target nets away from optimizer
        for p in self.actor_target.parameters():
            p.requires_grad = False

        for p in self.critic_target.parameters():
            p.requires_grad = False
        
        self.train_iter = 0

        if pretrained:
            try:
                self.load_weights(pretrained)
            except:
                print(f'ERROR: Could not load weights from {pretrained}')

    def _update_actor_target_net(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.update_rate + target_param.data * (1.0 - self.update_rate))

    def _update_critic_target_net(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.update_rate + target_param.data * (1.0 - self.update_rate))

    def save_weights(self, filepath):
        torch.save(self.actor.state_dict(), filepath + "_actor")
        torch.save(self.critic.state_dict(), filepath + "_critic")
    
    def load_weights(self, filepath):
        self.actor.load_state_dict(torch.load(filepath + "_actor"))
        self.actor.eval()
        self.critic.load_state_dict(torch.load(filepath + "_critic"))
        self.critic.eval()
    
    def name(self):
        return "DDPG"

    def store_transition(self, transition):
        self.buffer.add_transition(transition)
    
    def act(self, obs, eps=0):
        obs = torch.from_numpy(obs).float().unsqueeze(0)
        if np.random.random() > eps:
            # uncomment for pendulum env
            #return self.actor.forward(obs).detach().numpy()[0,0]
            return self.actor.forward(obs).detach().numpy().flatten()
        else:
            return self.env.action_space.sample()[:self.action_dim]
        



    def train(self, iter_fit=32):
        losses = []
        critic_l = 0
        actor_l = 0

        for i in range(iter_fit):
            # sample from the replay buffer
            data=self.buffer.sample(batch=self._config['batch_size'])
            done = np.stack(data[:,4])[:,None]
            s = np.stack(data[:,0]) # s_t
            s_prime = np.stack(data[:,3]) # s_t+1
            a = np.stack(data[:,1]) # a
            rew = np.stack(data[:,2])[:,None] # rew

            a = torch.FloatTensor(a)
            s = torch.FloatTensor(s)
            s_prime = torch.FloatTensor(s_prime)
            rew = torch.FloatTensor(rew)

            gamma=self.discount
            q = self.critic.forward(s, a)
            a_prime = self.actor_target.forward(s_prime)
            next_q = self.critic_target.forward(s_prime, a_prime.detach())
            td_target = rew + gamma * torch.logical_not(torch.from_numpy(done)) * next_q
            
            actor_loss = -self.critic.forward(s, self.actor.forward(s)).mean()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            critic_loss = self.critic.fit(q, td_target)
            critic_l += critic_loss
            actor_l += actor_loss

            losses.append(actor_loss + critic_loss)

            self.train_iter+=1
            if self.train_iter % self._config["update_target_every"] == 0:
                self._update_actor_target_net()
                self._update_critic_target_net()
            
        return losses




class OUNoise(object):
    def __init__(self, action_dim, action_low, action_high, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_dim
        self.low          = action_low
        self.high         = action_high
        self.reset()
        
    def reset(self, max_sigma=0.3, min_sigma=0.3):
        self.state = np.ones(self.action_dim) * self.mu
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)