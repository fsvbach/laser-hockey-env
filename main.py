#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:33:17 2021

@author: fsvbach
"""

from DQN import agent, training
from gym.spaces.discrete import Discrete
import laserhockey.hockey_env as h_env
from laserhockey.gameplay import gameplay
import matplotlib.pyplot as plt
from DDPG import train as ddpg_train
from DDPG.ddpg_agent import DDPGAgent

name='defense'
mode=2


load_weights=False
store_weights='basic_opponent'

env = h_env.HockeyEnv(mode=mode)
player2 = h_env.BasicOpponent()

q_agent = agent.DQNAgent(env.observation_space, 
                         env.discrete_action_space,
                        convert_func =  env.discrete_to_continous_action,
                        pretrained   = f'DQN/weights/{load_weights}')


ddpg_player = DDPGAgent(env.observation_space, 
                         env.action_space)                

player2 = h_env.BasicOpponent()


losses, rewards = training.train(env, q_agent, player2=player2, name=store_weights, max_episodes=50000)

plt.plot(training.running_mean(losses,64))
plt.savefig(f'Plots/{store_weights}_losses')
plt.show()
plt.close()

plt.plot(training.running_mean(rewards,64))
plt.savefig(f'Plots/{store_weights}_rewards')
plt.show()
plt.close()

# losses, rewards = training.train(env,
#                                  q_agent, 
#                                  player2=player2, 
#                                  name=name+'_extend', 
#                                  max_episodes=30000)

stats = gameplay(env, q_agent, player2=False, N=10, show=True, analyze=False)
print(stats)

env.close()
