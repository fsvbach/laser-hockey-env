import time
import numpy as np
import torch
from .ddpg_agent import OUNoise

def train(env, agent, player2=False, max_episodes=200, max_steps=200, show=False, name='test'):

    ob = env.reset()
    noise = OUNoise(env.action_space)
    
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
        episode_losses = 0
        for step in range(max_steps):
            
            done = False
            act = agent.act(ob)
            act = noise.get_action(act, step)
            act2 = [0,0.,0,0] 

            if player2:
                act2 = player2.act(obs2)

            (ob_new, reward, done, _info) = env.step(np.hstack([act,act2]))
            total_reward += reward
            agent.store_transition((ob, act, reward, ob_new, done))      
            if agent.buffer.size > agent._config["batch_size"]:
                episode_losses += np.sum(agent.train(1))
            
            ob=ob_new
            obs2 = env.obs_agent_two()

            if show:
                time.sleep(1.0/fps)
                env.render(mode='human')

            if done: break    

        losses.append(episode_losses)
        stats.append([i,total_reward,step+1])    
    
        if ((i-1)%20==0):
            print("{}: Done after {} steps. Reward: {}".format(i, step+1, total_reward))
    
    agent.save_weights(f'DDPG/weights/{name}')

    return losses, [r[1] for r in stats]
    
