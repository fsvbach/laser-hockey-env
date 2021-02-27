#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 13:33:27 2021

@author: fsvbach
"""


import numpy as np
import laserhockey.hockey_env as h_env
import pylab as plt
import time

max_len = 500

env = h_env.HockeyEnv()

player1 = h_env.BasicOpponent()
player2 = h_env.BasicOpponent()
# player2 = h_env.HumanOpponent(env=env, player=2)
player1 = h_env.HumanOpponent(env=env, player=1)

obs = env.reset()

obs_buffer = []
reward_buffer=[]

env.render()
time.sleep(1)
obs_agent2 = env.obs_agent_two()
for _ in range(max_len):
    time.sleep(0.2)
    env.render()
    a1 = player1.act(obs) 
    a2 = player2.act(obs_agent2)
    obs, r, d, info = env.step(np.hstack([a1,a2]))    
    obs_agent2 = env.obs_agent_two()
    
    obs_buffer.append(obs_agent2)
    reward_buffer.append(r)
    if d: break

env.close()

plt.plot(reward_buffer[:-1])