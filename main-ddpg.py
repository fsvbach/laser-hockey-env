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

env = h_env.HockeyEnv()
attack = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
defense = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
normal = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)

<<<<<<< HEAD
name='ddpg-normal-noeps-noise-td3-40000'

td3 = TD3(18, 4, 1.0, env)
td3.load(filename='stronger')

ddpg_player = DDPGAgent(env,
                         actor_lr=1e-4,
                         critic_lr=1e-3,
                         update_rate=0.05,
                         discount=0.9, update_target_every=20)

losses, rewards = ddpg_train.train(normal, ddpg_player, player2=td3, name=name, max_episodes=40000, show=False)
=======
load_weights = False
store_weights = 'strong_basic_oppponent'

player2 = h_env.BasicOpponent(weak=True)


losses, rewards = training.train(normal, ddpg_player, player2=player2, name=name, max_episodes=10000, show=False)
>>>>>>> cc1734c9aa79a79bca0990e3e35383b4451ccc2e

plt.plot(training.running_mean(losses,64))
plt.savefig(f'Plots/{name}_losses')
plt.show()
plt.close()

plt.plot(training.running_mean(rewards,64))
plt.savefig(f'Plots/{name}_rewards')
plt.show()
<<<<<<< HEAD
plt.close()
=======
plt.close() 
>>>>>>> cc1734c9aa79a79bca0990e3e35383b4451ccc2e
  

##for i in range (20):
stats = gameplay(normal, ddpg_player, player2=td3, N=100, show=False, analyze=False)
print(stats)

defense.close()
attack.close()
env.close()
