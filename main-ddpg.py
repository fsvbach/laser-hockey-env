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

name='ddpg-normal-params7-25000'


ddpg_player = DDPGAgent(env,
                         actor_lr=1e-4,
                         critic_lr=1e-3,
                         update_rate=0.05,
                         discount=0.9, update_target_every=20)

losses, rewards = ddpg_train.train(normal, ddpg_player, player2=False, name=name, max_episodes=1000, show=False)

plt.plot(training.running_mean(losses,64))
plt.savefig(f'Plots/{name}_losses')
plt.show()
plt.close()

plt.plot(training.running_mean(rewards,64))
plt.savefig(f'Plots/{name}_rewards')
plt.show()
plt.close()
  
""" ddpg_player = params4 = DDPGAgent(env.observation_space, 
                         env.action_space,
                         actor_lr=1e-5,
                         critic_lr=1e-3,
                         update_rate=0.01,
                         discount=0.9, update_target_every=5, pretrained='DDPG/weights/ddpg-attack-params5') """

# player2 = h_env.BasicOpponent()
stats = gameplay(attack, ddpg_player, player2=False, N=100, show=False, analyze=False)
print(stats)

defense.close()
attack.close()
env.close()


def grid_search(param_grid):
    for i, params in enumerate(param_grid):

        ddpg_player = DDPGAgent(env.observation_space, 
                            env.action_space,
                            actor_lr=params[0],
                            critic_lr=params[1],
                            update_rate=params[2],
                            discount=params[3])     

        losses, rewards = ddpg_train.train(attack, ddpg_player, player2=False, name=name, max_episodes=400, show=False)
        #losses.append(loss)
        #rewards.append(rewards)

        plt.plot(training.running_mean(losses,64))
        plt.savefig(f'Plots/{name}_params{i}_losses')
        #plt.show()
        #plt.close()

        plt.plot(training.running_mean(rewards,64))
        plt.savefig(f'Plots/{name}_params{i}_rewards')
        #plt.show()
        #plt.close()

        # player2 = h_env.BasicOpponent()
        stats = gameplay(attack, ddpg_player, player2=False, N=100, show=False, analyze=True)
        print("params ", i, ": ", stats)