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

env = h_env.HockeyEnv()

basic   = h_env.BasicOpponent(weak=False)

q_agent = agent.DQNAgent(env.observation_space, 
                         env.discrete_action_space,
                        convert_func =  env.discrete_to_continous_action,
                        pretrained   = f'DQN/weights/defense')

ddpg    = DDPGAgent(env,
                         actor_lr=1e-4,
                         critic_lr=1e-3,
                         update_rate=0.05,
                         discount=0.9, update_target_every=20,
                         pretrained='DDPG/weights/ddpg-attack-ounoise-5001')

td3     = TD3(env.observation_space.shape[0],
              env.num_actions)
td3.load('best_avg')

td4     = TD3(env.observation_space.shape[0],
              env.num_actions)
td4.load('stronger')


stats = gameplay(env, td3, player2=td4, N=10, show=True, analyze=False)
print(stats)

env.close()
