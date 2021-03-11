#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:16:26 2021

@author: johannes
"""

from DQN import agent, training
from gym.spaces.discrete import Discrete
import laserhockey.hockey_env as h_env
from laserhockey.gameplay import gameplay
import matplotlib.pyplot as plt
from DDPG import train as ddpg_train
from DDPG.ddpg_agent import DDPGAgent
from TD3.agent import TD3

env = h_env.HockeyEnv()
attack = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
defense = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
normal = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)

load_weights = 'weak_basic_opponent'
store_weights = 'weak_basic_oppponent'

td3 = TD3(18, 4, 1.0, env)
td3.load(filename='stronger')
player2 = h_env.BasicOpponent(weak=False)


q_agent = agent.DQNAgent(env.observation_space, env.discrete_action_space,
                        convert_func =  env.discrete_to_continous_action,
                        pretrained   = f'DQN/weights/{load_weights}')

#losses, rewards = training.train(normal, q_agent, player2=player2, name=store_weights, show=False, max_episodes=10000)


stats = gameplay(normal, q_agent, player2=td3, N=20, show=True, analyze=False)
print(stats)

defense.close()
attack.close()
env.close()
