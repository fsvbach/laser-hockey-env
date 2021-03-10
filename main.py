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
from TD3.agent import TD3

name='gameplay'
mode=0


env = h_env.HockeyEnv(mode=mode)
player2 = h_env.BasicOpponent()

q_agent = agent.DQNAgent(env.observation_space, 
                         env.discrete_action_space,
                        convert_func =  env.discrete_to_continous_action,
                        pretrained   = f'DQN/weights/{name}')

td3 = TD3(18, 4, 1.0, env)
td3.load(filename='next')

# losses, rewards = training.train(env,
#                                  q_agent, 
#                                  player2=player2, 
#                                  name=name, 
#                                  max_episodes=50000)


stats = gameplay(env, q_agent, player2=td3, N=5, show=True, analyze=False)
print(stats)

env.close()
