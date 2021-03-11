#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 18:11:13 2021

@author: johannes
"""

from laserhockey import HockeyEnv
import numpy as np

class TrainingHall(HockeyEnv):
    
  def register_opponents (self, agents): 
      self.agents = agents
      
  def step(self, action):
    ob2 = self.obs_agent_two()
    a2 = self.opponent.act(ob2)
    action2 = np.hstack([action, a2])
    return super().step(action2)
      
  def reset(self): 
      super().reset()
      self.opponent = np.random.sample(self.agents)
      
      
      