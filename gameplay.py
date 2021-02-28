#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 13:33:27 2021

@author: fsvbach
"""

from DQN import agent
from gym.spaces.discrete import Discrete
import numpy as np
import laserhockey.hockey_env as h_env
import time

max_len = 500
fps = 50

env = h_env.HockeyEnv()

ac_space = Discrete(8)
o_space = env.observation_space

player2 = h_env.BasicOpponent()
player1 = h_env.HumanOpponent(env=env, player=1)
player1 = agent.DQNAgent(o_space, ac_space)

player1.load_weights('DQN/weights/test')

obs = env.reset()

obs_buffer = []
reward_buffer=[]

env.render()
time.sleep(1/fps)
obs_agent2 = env.obs_agent_two()
for _ in range(max_len):
    time.sleep(1/fps)
    env.render()
    a1 = player1.act(obs) 
    a2 = player2.act(obs_agent2)
    obs, r, d, info = env.step(np.hstack([env.discrete_to_continous_action(a1),a2]))    
    obs_agent2 = env.obs_agent_two()
    
    obs_buffer.append(obs_agent2)
    reward_buffer.append(r)
    if d: break

env.close()



'''
TODO: 
    rename files (main -> train agent)
    in gameplay: load weights
    in agent: integrate buffer, always return continuous actions

'''
