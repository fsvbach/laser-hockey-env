#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 18:11:13 2021

@author: johannes
"""

import laserhockey.hockey_env as h_env
import numpy as np


class TrainingHall(h_env.HockeyEnv):

  def __init__(self, mode=h_env.HockeyEnv.NORMAL, weak_opponent=False):
    self.opponent = h_env.BasicOpponent(weak=weak_opponent)
    self.agents = np.array([self.opponent])
    super().__init__(mode=mode, keep_mode=True)
    

  def register_opponents (self, opponents): 
      self.agents = np.append(self.agents, opponents)
      
  def step(self, action):
      ob2 = self.obs_agent_two()
      a2 = self.opponent.act(ob2)
      action2 = np.hstack([action, a2])
      return super().step(action2)
      
  def reset(self, one_starting=None, mode=None): 
      self.opponent = np.random.choice(self.agents)
      return super().reset(one_starting, mode)
      
      
      
      