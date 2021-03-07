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
from DDPG import train as ddpg_train
from DDPG.ddpg_agent import DDPGAgent

env = h_env.HockeyEnv()
attack = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
defense = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
play = h_env.HockeyEnv_BasicOpponent()

q_agent = agent.DQNAgent(env.observation_space, 
                         Discrete(8),
                         #convert_func =  env.discrete_to_continous_action, #only in gameplay mode!!
                        pretrained   = 'DQN/weights/defense')

player1 = agent.DQNAgent( env.observation_space, 
                          Discrete(8), 
                           eps=0,
                          convert_func =  env.discrete_to_continous_action,
                          pretrained   = 'DQN/weights/defense2000')

ddpg_player = DDPGAgent(env.observation_space, 
                         env.action_space)                

losses, rewards = train.train(attack, q_agent, player2=False, name='shootdefense')
#losses, rewards = ddpg_train.train(attack, ddpg_agent, player2=False, name='shootdefense')
plt.plot(losses)
plt.show()
plt.plot(rewards)
plt.show()

player2 = h_env.BasicOpponent()
stats = gameplay(attack, ddpg_player, player2=player2, N=5, show=True)
print(stats)

defense.close()
attack.close()
env.close()
play.close()
