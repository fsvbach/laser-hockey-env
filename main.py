#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:33:17 2021

@author: fsvbach
"""

from laserhockey.hockey_env import HockeyEnv, BasicOpponent, StupidOpponent
from laserhockey.gameplay import gameplay
from DQN import agent
from DDPG.ddpg_agent import DDPGAgent
from TD3.agent import TD3

env = HockeyEnv()

stupid = StupidOpponent()
basic = BasicOpponent(weak=False)
weak = BasicOpponent(weak=True)

dqn = agent.DQNAgent(env.observation_space, 
                         env.discrete_action_space,
                        convert_func =  env.discrete_to_continous_action,
                        pretrained   = 'DQN/weights/training_hall_1')

ddpg = DDPGAgent(env,
                actor_lr=1e-4,
                critic_lr=1e-3,
                update_rate=0.05,
                discount=0.9, update_target_every=20,
                pretrained='DDPG/weights/ddpg-normal-eps-noise-basic-35000')

td3 = TD3(pretrained='best_avg')

stats = gameplay(env, td3, player2=weak, N=10, show=True, analyze=True)
print(stats)

env.close()
