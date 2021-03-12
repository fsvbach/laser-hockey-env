#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:33:17 2021

@author: fsvbach
"""

from DQN import agent, training
from gym import spaces
import numpy as np
import laserhockey.hockey_env as h_env
from laserhockey.gameplay import gameplay
import matplotlib.pyplot as plt
from DDPG import train as ddpg_train
from DDPG.ddpg_agent import DDPGAgent
from TD3.agent import TD3, Trainer, ReplayBuffer, train, observe

env = h_env.HockeyEnv(mode=0)

basic   = h_env.BasicOpponent(weak=True)

# q_agent = agent.DQNAgent(env.observation_space, 
#                          env.discrete_action_space,
#                         convert_func =  env.discrete_to_continous_action,
#                         pretrained   = f'DQN/weights/defense')

ddpg    = DDPGAgent(env,
                         actor_lr=1e-4,
                         critic_lr=1e-3,
                         update_rate=0.05,
                         discount=0.9, update_target_every=20,
                         pretrained='DDPG/weights/ddpg-normal-eps-noise-basic-35000')

td3     = TD3(env.observation_space.shape[0],
              env.num_actions)
td3.load('TDweak')

td4     = TD3(env.observation_space.shape[0],
              env.num_actions)
td4.load('strong')

td5     = TD3(env.observation_space.shape[0],
              env.num_actions)
td5.load('best_avg')


stats = gameplay(env, td5, player2=basic, N=100, show=False, analyze=False)
print(stats)

env.close()
