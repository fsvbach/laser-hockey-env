import torch
import numpy as np
from .memory import Memory
from .actor import Actor
from .critic import Critic

class DDPGAgent(object):
    """
    Agent implementing DDPG.
    """
    def __init__(self, observation_space, action_space, pretrained=False, **userconfig):

        self._observation_space = observation_space
        self._action_space = action_space
        self._config = {
            "discount": 0.99,
            "buffer_size": int(1e5),
            "batch_size": 128,
            "actor_lr": 0.0001,
            "critic_lr": 0.001,
            "update_target_every": 20,
            "actor_update_rate": 0.001,
            "critic_update_rate": 0.001
        }
        self._config.update(userconfig)

        self.buffer = Memory(max_size=self._config["buffer_size"])

        self.actor_update_rate = self._config["actor_update_rate"]
        self.critic_update_rate = self._config["critic_update_rate"]

        # Actor
        self.actor = Actor(action_dim=action_space.shape[0],
                           observation_dim=self._observation_space.shape[0],
                           learning_rate = self._config["actor_lr"])

        self.actor_target = Actor(action_dim=action_space.shape[0],
                           observation_dim=self._observation_space.shape[0],
                           learning_rate=0)

        # Critic
        self.critic = Critic(action_dim=action_space.shape[0],
                             observation_dim=self._observation_space.shape[0],
                             learning_rate=self._config["critic_lr"])

        self.critic_target = Critic(action_dim=action_space.shape[0],
                             observation_dim=self._observation_space.shape[0],
                             learning_rate=0)

        if pretrained:
            try:
                self.actor.load_state_dict(torch.load(pretrained + "_actor"))
                self.actor.eval()
                self.critic.load_state_dict(torch.load(torch.load(pretrained + "_critic")))
                self.critic.eval()
            except:
                print(f'ERROR: Could not load weights from {pretrained}')
        

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


    def _update_actor_target_net(self):
        with torch.no_grad():
            for p1, p2 in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_updated = torch.mul(p1, (1 - self.actor_update_rate))
                target_updated.add_(torch.mul(p2, self.actor_update_rate))
                p1.copy_(target_updated)

    def _update_critic_target_net(self):
        with torch.no_grad():
            for p1, p2 in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_updated = torch.mul(p1, (1 - self.critic_update_rate))
                target_updated.add_(torch.mul(p2, self.critic_update_rate))
                p1.copy_(target_updated)

    def save_weights(self, filepath):
        torch.save(self.actor.state_dict(), filepath +  "_actor")
        torch.save(self.critic.state_dict(), filepath + "_critic")

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def act(self, obs):
        return self.actor.get_action(obs).detach()

    def train(self, iter_fit=32):
        losses = []

        self.train_iter+=1
        if self.train_iter % self._config["update_target_every"] == 0:
            self._update_actor_target_net()
            self._update_critic_target_net()

        for i in range(iter_fit):
            # sample from the replay buffer
            data=self.buffer.sample(batch=self._config['batch_size'])
            # column_stack for alignment on dim 2

            #s = np.column_stack(data[:,0]).reshape(self._config['batch_size'], -1)
            #a = torch.vstack(data[:, 1].tolist()) # [batch_size,1] tensor
            #rew = np.vstack(data[:,2])
            #s_prime = np.stack(data[:,3]).reshape(self._config['batch_size'], -1)
            done = np.stack(data[:,4])[:,None]
            s = np.stack(data[:,0]) # s_t
            s_prime = np.stack(data[:,3]) # s_t+1
            a = np.stack(data[:,1]) # a
            rew = np.stack(data[:,2])[:,None].flatten() # rew

            gamma=self._config['discount']
            a_prime = self.actor_target.get_action(s_prime)

            td_target = torch.from_numpy(rew) + gamma * torch.logical_not(torch.from_numpy(done)) * self.critic_target.Q_value(
                torch.from_numpy(s_prime), a_prime)

            q = self.critic.Q_value(torch.from_numpy(s), torch.from_numpy(a))
            critic_fit_loss = self.critic.fit(q, td_target)
            #TODO: why actions only non-negative?
            #TODO: Initialization of critic: --> 1 problem due to this!
            an = self.actor.get_action(s)
            q_policy = self.critic.Q_value(torch.from_numpy(s).detach(), an)

            actor_fit_loss = self.actor.fit(q_policy)

            losses.append([actor_fit_loss, critic_fit_loss])

            

        return losses
