#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:33:17 2021

@author: fsvbach
"""

from DQN import agent, train
from gym.spaces.discrete import Discrete
import laserhockey.hockey_env as h_env
from laserhockey.gameplay import gameplay
import matplotlib.pyplot as plt

env = h_env.HockeyEnv()
attack = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
defense = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
play = h_env.HockeyEnv_BasicOpponent()


q_agent = agent.DQNAgent(env.observation_space, 
                         Discrete(8),
                         #convert_func =  env.discrete_to_continous_action, #only in gameplay mode!!
                        pretrained   = 'DQN/weights/defense')


# losses, rewards = train.train(defense, 
#                               q_agent, 
#                               player2=False, 
#                               name='defense2000', 
#                               max_episodes=1000)
# plt.plot(losses)
# plt.show()
# plt.plot(rewards)
# plt.show()


player2 = h_env.BasicOpponent()
player1 = agent.DQNAgent( env.observation_space, 
                          Discrete(8), 
                           eps=0,
                          convert_func =  env.discrete_to_continous_action,
                          pretrained   = 'DQN/weights/defense2000')

stats = gameplay(defense, player1, player2=False, N=10, show=True)
print(stats)

defense.close()