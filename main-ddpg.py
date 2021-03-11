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

env = h_env.HockeyEnv()
attack = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
defense = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
normal = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)

load_weights = False
store_weights = 'strong_basic_oppponent'

player2 = h_env.BasicOpponent(weak=True)


losses, rewards = training.train(normal, ddpg_player, player2=player2, name=name, max_episodes=10000, show=False)

plt.plot(training.running_mean(losses,64))
plt.savefig(f'Plots/{name}_losses')
plt.show()
plt.close()

plt.plot(training.running_mean(rewards,64))
plt.savefig(f'Plots/{name}_rewards')
plt.show()
plt.close() 
  

##for i in range (20):
stats = gameplay(normal, ddpg_player, player2=player2, N=100, show=True, analyze=False)
print(stats)

defense.close()
attack.close()
env.close()
