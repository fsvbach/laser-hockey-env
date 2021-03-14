
import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddpg_agent import OUNoise
import time
from ddpg_agent import DDPGAgent

env = gym.make("Pendulum-v0")
name = "ddpg-pendulum-v0"

agent = DDPGAgent(env)
noise = OUNoise(action_dim=env.action_space.shape[0], 
                        action_low=env.action_space.low, 
                        action_high=env.action_space.high,
                        max_sigma=0.3, min_sigma=0.2)
rewards = []
avg_rewards = []
batch_size=128
max_episodes=400
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

agent.save_weights(f'DDPG/weights/{name}')

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig(f'Plots/ddpg_pendulum-new_rewards')
plt.show()