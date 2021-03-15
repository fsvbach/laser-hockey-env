#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:33:17 2021

@author: fsvbach
"""

from laserhockey.hockey_env import HockeyEnv, BasicOpponent, StupidOpponent
from laserhockey.gameplay import gameplay
from laserhockey.TrainingHall import TrainingHall
from DQN import agent
from DDPG.ddpg_agent import DDPGAgent
from TD3.agent import TD3

env = HockeyEnv()

stupid = StupidOpponent()
basic = BasicOpponent(weak=False)
weak = BasicOpponent(weak=True)

# env = TrainingHall(weak_opponent=True)

basic  = BasicOpponent(weak=False)


# env.register_opponents([basic,last])#,ddpg,q_agent])

td3 = TD3(pretrained='superagent')
td4 = TD3(pretrained='best_avg')

stats = gameplay(env, td3, player2=td4, N=10, show=True, analyze=False)
print(stats)

# env.close()
