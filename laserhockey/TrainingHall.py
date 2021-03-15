#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 18:11:13 2021

@author: johannes
"""

import laserhockey.hockey_env as h_env
from gym import spaces
import numpy as np

def softmax(x):
    x -= np.max(x)
    return np.exp(x)/np.sum(np.exp(x))

class TrainingHall(h_env.HockeyEnv):

  def __init__(self, mode=h_env.HockeyEnv.NORMAL, weak_opponent=True):
    self.opponent = h_env.BasicOpponent(weak=weak_opponent)
    self.agents = np.array([self.opponent])
    self.stats = {self.opponent: [0]}
    self.weights = []
    self.counter = 0
    self.period  = 100
    super().__init__(mode=mode, keep_mode=True)
    self.action_space = spaces.Box(-1, +1, (4,), dtype=np.float32)
    
  def register_opponents (self, opponents): 
      self.agents = np.append(self.agents, opponents)
      for agent in opponents:
          self.stats[agent] =  [0]
      self.weights=[]
      self.update_weights()
      self.counter = 0
      
  def step(self, action):
      ob2 = self.obs_agent_two()
      a2 = self.opponent.act(ob2)
      action2 = np.hstack([action, a2])
      return super().step(action2)
      
  def reset(self, one_starting=None, mode=None): 
      self.stats[self.opponent].append(self.winner)
      if self.counter % self.period == 0:
          self.counter=0
          weights = self.update_weights(add=True)
          print(f"\ncurrent winratios: {self.weights[-1]}")
          print(f'... with weights: {weights}\n')
          self.next_opponents = np.random.choice(self.agents, 
                                                 size=self.period,
                                                 p=weights)
      self.opponent = self.next_opponents[self.counter]
      self.counter += 1
      return super().reset(one_starting, mode)
  
  def update_weights(self, add=False):
      ratios = np.zeros_like(self.agents, dtype='float64')
      for i, agent in enumerate(self.agents):
          stats = self.stats[agent]
          if len(stats) > 100:
              stats = stats[-100:]
          ratios[i] = sum(stats)/len(stats)
      if add:
          self.weights.append(ratios)
      return softmax(-ratios)
  
      
      
      
      