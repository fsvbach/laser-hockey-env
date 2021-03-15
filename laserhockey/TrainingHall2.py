#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:40:57 2021

@author: johannes
"""

import laserhockey.hockey_env as h_env
import numpy as np


class TrainingHall2(h_env.HockeyEnv):
      
    def __init__(self, mode=h_env.HockeyEnv.NORMAL, weak_opponent=True):
      self.opponent = h_env.BasicOpponent(weak=weak_opponent)
      self.agents = np.array([self.opponent])
      self.counter = 0
      self.n = 1
      super().__init__(mode=mode, keep_mode=True)
      
    def step(self, action):
        ob2 = self.obs_agent_two()
        a2 = self.opponent.act(ob2)
        action2 = np.hstack([action, a2])
        return super().step(action2)
    
    def register_opponents (self, opponents): 
        self.agents = np.append(self.agents, opponents)
        self.n += len(opponents)
        
    def reset(self, one_starting=None, mode=None):
        self.opponent = self.agents[self.counter]
        self.counter = (self.counter + 1) % self.n
        return super().reset(one_starting, mode)