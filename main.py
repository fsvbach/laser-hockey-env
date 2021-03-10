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
normal = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)

ddpg2 = DDPGAgent(env,
                         actor_lr=1e-4,
                         critic_lr=1e-3,
                         update_rate=0.05,
                         discount=0.9, update_target_every=20,
                         pretrained='DDPG/weights/ddpg-attack-ounoise-5001')

ddpg_agent = DDPGAgent(env,
                         actor_lr=1e-4,
                         critic_lr=1e-3,
                         update_rate=0.05,
                         discount=0.9, update_target_every=20,
                         pretrained='DDPG/weights/ddpg-normal-eps-noise-10000')

td3 = TD3(18, 4, 1.0, env)
td3.load(filename='next')

# losses, rewards = training.train(env,
#                                  q_agent, 
#                                  player2=player2, 
#                                  name=name, 
#                                  max_episodes=50000)


stats = gameplay(normal, ddpg_agent, player2=ddpg2, N=100, show=False, analyze=False)
print(stats)

env.close()
