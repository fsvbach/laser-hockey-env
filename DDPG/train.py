import time
import numpy as np
import torch
from .ddpg_agent import OUNoise
import sys

def train(env, agent, player2=False, max_episodes=200, max_steps=200, show=False, name='test'):

    ob = env.reset()
    if player2:
        # manually make action dim = 8 (otherwise only 4 for one player)
        noise = OUNoise(action_dim= 2 * env.action_space.shape[0], action_low=np.tile(env.action_space.low, 2), action_high=np.tile(env.action_space.high, 2))
    else:
        noise = OUNoise(action_dim=env.action_space.shape[0], action_low=env.action_space.low, action_high=env.action_space.high)

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
        ob2 = env.reset()
        episode_losses = 0
        for step in range(max_steps):
            
            done = False
            act = agent.act(ob)
            act = noise.get_action(act, step)
            act2 = [0,0.,0,0] 

            # if two players only add noise to first 4 action elements
            if player2:
                act2 = player2.act(ob2)

            (ob_new, reward, done, _info) = env.step(np.hstack([act,act2]))
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

