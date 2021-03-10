import time
import numpy as np
import torch
from .ddpg_agent import OUNoise
import sys

def train(env, agent, player2=False, max_episodes=200, max_steps=200, show=False, name='test'):

    ob = env.reset()
    
    noise = OUNoise(action_dim=env.num_actions, 
                        action_low=env.action_space.low[:4], 
                        action_high=env.action_space.high[:4],
                        max_sigma=0.3, min_sigma=0.2)

    stats = []
    losses = []
    rewards = []
    avg_rewards = []
    fps=50
    show = False
    eps = 1
    cnt = 1

    for i in range(max_episodes):
        noise.reset()
        ob = env.reset()
        ob2 = env.reset()
        episode_losses = 0
        total_reward = 0

        for step in range(max_steps):
            done = False
            act = agent.act(ob, eps=0)
            act = noise.get_action(act, step)
            act2 = [0,0.,0,0]

            # if two players only add noise to first 4 action elements
            if player2:
                act2 = player2.act(ob2)

            (ob_new, reward, done, _info) = env.step(np.hstack([act,act2]))
            reward = reward + _info["winner"] + _info["reward_closeness_to_puck"] + _info["reward_touch_puck"] + _info["reward_puck_direction"]
            + _info["reward_puck_before"]

            total_reward += reward
            agent.store_transition((ob, act, reward, ob_new, done))      
            
            ob=ob_new
            ob2 = env.obs_agent_two()

            if show:
                time.sleep(1.0/fps) 
                env.render(mode='human')

            if done: break    

        losses.extend(agent.train(32))
        stats.append([i,total_reward,step+1])    
    
        if ((i-1)%20==0):
            print("{}: Done after {} steps. Reward: {}".format(i, step+1, total_reward))
    
    agent.save_weights(f'DDPG/weights/{name}')

    return losses, [r[1] for r in stats]


def alt_train(env, agent, player2=False, max_episodes=200, max_steps=200, show=False, name='test'):
    ob = env.reset()
    noise = OUNoise(env.action_space)
    rewards = []
    avg_rewards = []
    losses = []

    for episode in range(max_episodes):
        state = env.reset()
        noise.reset()
        episode_reward = 0
        episode_loss = 0
    
        for step in range(max_steps):
            
            done = False
            act = agent.act(ob)
            act = noise.get_action(act, step)
            act2 = [0,0.,0,0] 

            if player2:
                act2 = player2.act(obs2)

            (ob_new, reward, done, _info) = env.step(np.hstack([act,act2]))
            agent.store_transition((ob, act, reward, ob_new, done))    

            ob = ob_new
            episode_reward += reward
        
            if show:
                time.sleep(1.0/50)
                env.render(mode='human')  

            if done:
                sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-20:])))
                break
        
        losses.append(agent.train(32))
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-20:]))

    return rewards, avg_rewards

