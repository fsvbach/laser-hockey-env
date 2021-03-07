import time
import numpy as np
import torch
from ddpg_agent import OUNoise

def train(env, ddpg_agent, player2=False, max_episodes=200, max_steps=200, show=False, name='test'):

    ob = env.reset()
    noise = OUNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))
    
    stats = []
    losses = []
    rewards = []
    avg_rewards = []
    fps=50
    show = False

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
    
