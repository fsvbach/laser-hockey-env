import time
import numpy as np
import torch


def train(env, ddpg_agent, player2=False, max_episodes=200, max_steps=200, show=False, name='test'):

    ob = env.reset()
    noise = OUNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))
    
    stats = []
    losses = []
    fps=50

    for i in range(max_episodes):
        #print("Starting a new episode")
        total_reward = 0
        noise.reset()
        ob = env.reset()
        for t in range(max_steps):
            done = False
        
            #a = ddpg_agent.actor.get_action(ob.T).detach().numpy().flatten() + noise()
            a = ddpg_agent._action_space.sample() + noise()
            a2 = [0,0.,0,0] 

            if player2:
                a2 = player2.act(obs2)

            (ob_new, reward, done, _info) = env.step(np.hstack([a,a2]))
            total_reward += reward
            ddpg_agent.store_transition((ob, torch.from_numpy(a), reward, ob_new, done))            
            ob=ob_new
            obs2 = env.obs_agent_two()
            if show:
                time.sleep(1.0/fps)
                env.render(mode='human')        
            if done: break    

        losses.extend(ddpg_agent.train(32))
        stats.append([i,total_reward,t+1])    
    
        if ((i-1)%20==0):
            print("{}: Done after {} steps. Reward: {}".format(i, t+1, total_reward))
    
    ddpg_agent.save_weights(f'DDPG/weights/{name}')

    return losses, [r[1] for r in stats]
    
class OUNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
