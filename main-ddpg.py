#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 14:33:17 2021

@author: fsvbach
"""

#from DQN import agent, training
from gym.spaces.discrete import Discrete
import laserhockey.hockey_env as h_env
from laserhockey.gameplay import gameplay
import matplotlib.pyplot as plt
from DDPG import train as ddpg_train
from DDPG.ddpg_agent import DDPGAgent
from TD3.agent import TD3
from DQN.agent import DQNAgent
import DQN.training as training

env = h_env.HockeyEnv()
attack = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
defense = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
normal = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)

weak = h_env.BasicOpponent(weak=True)

strong = h_env.BasicOpponent(weak=False)

#td4     = TD3(env.observation_space,
 #             env.num_actions)
#td4.load('stronger')

# best runs
# win to strong:
# ddpg-normal-eps-noise-basic-35000
# lose to strong:
# ddpg-normal-noeps-noise-td3-40000
# ddpg-attack-ounoise-5001
# ddpg-normal-eps-noise-10000


ddpg_trained= DDPGAgent(env,
                         actor_lr=1e-4,
                         critic_lr=1e-3,
                         update_rate=0.02,
                         discount=0.9, update_target_every=20, pretrained="DDPG/weights/ddpg-normal-eps-noise-basic-35000")


ddpg_player = DDPGAgent(env,
                         actor_lr=1e-5,
                         critic_lr=1e-4,
                         update_rate=0.01,
                         discount=0.9, update_target_every=20)


""" name="ddpg-noise-eps-normal-strong-10000"
losses, rewards = ddpg_train.train(normal, ddpg_player, player2=strong, name=name, max_episodes=10000, show=False, reward_weights=[20, 1, 10, 0, 2])


plt.plot(training.running_mean(losses,64))
plt.savefig(f'Plots/{name}_losses')
plt.show()
plt.close()

plt.plot(training.running_mean(rewards,64))
plt.savefig(f'Plots/{name}_rewards')
plt.show()
plt.close() """

  

for i in range (20):
    stats = gameplay(normal, ddpg_trained, player2=strong, N=20, show=True, analyze=False)
    print(stats)

defense.close()
attack.close()
env.close()
