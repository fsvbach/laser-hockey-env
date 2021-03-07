
import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddpg_agent import OUNoise
import time
from ddpg_agent import DDPGAgent

env = gym.make("Pendulum-v0")

agent = DDPGAgent(env.observation_space, env.action_space)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []
max_episodes=100
max_steps=500
show=False

for episode in range(max_episodes):
    state = env.reset()
    noise.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        action = agent.act(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action) 

        agent.store_transition((state, action, reward, new_state, done))
        if agent.buffer.size > batch_size:  
            agent.train(1)

        state = new_state
        episode_reward += reward
        
        if show:
            time.sleep(1.0/50)
            env.render(mode='human')  

        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig(f'Plots/ddpg_pendulum-v0_rewards')
plt.show()