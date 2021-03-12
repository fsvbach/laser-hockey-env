#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:16:26 2021

@author: johannes
"""

from DQN import agent, training
from DQN.genetic_optimization import GeneticOptimization
import laserhockey.hockey_env as h_env
from laserhockey.gameplay import gameplay
from DDPG.ddpg_agent import DDPGAgent
from TD3.agent import TD3
from laserhockey.TrainingHall import TrainingHall

env = h_env.HockeyEnv()
attack = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
defense = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
normal = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
training_hall = TrainingHall()

load_weights = False
store_weights = 'training_hall_2'


td3 = TD3(18,4,1.0)
td3.load('stronger')

basic_opponent = h_env.BasicOpponent(weak=False)
ddpg = DDPGAgent(training_hall,
                         actor_lr=1e-4,
                         critic_lr=1e-3,
                         update_rate=0.05,
                         discount=0.9, update_target_every=20,
                         pretrained='DDPG/weights/ddpg-normal-eps-noise-10000')

ddpg2 = DDPGAgent(training_hall,
                         actor_lr=1e-4,
                         critic_lr=1e-3,
                         update_rate=0.05,
                         discount=0.9, update_target_every=20,
                         pretrained='DDPG/weights/ddpg-normal-eps-noise-basic-35000')

q_agent = agent.DQNAgent(training_hall.observation_space, training_hall.discrete_action_space,
                        convert_func =  env.discrete_to_continous_action,
                        pretrained   = f'DQN/weights/{load_weights}')

q_agent2 = agent.DQNAgent(training_hall.observation_space, training_hall.discrete_action_space,
                        convert_func =  env.discrete_to_continous_action,
                        pretrained   = 'DQN/weights/training_hall_1')

training_hall.register_opponents([ddpg, td3, basic_opponent, q_agent2, ddpg2])

#GeneticOptimization(10, 25, 100, env=training_hall).run()

#losses, rewards = training.train(normal, q_agent, player2=basic_opponent, name=store_weights, show=False, max_episodes=10000)
losses, rewards = training.train(training_hall, q_agent, name=store_weights, show=False, max_episodes=20000)

stats = gameplay(training_hall, q_agent, player2=td3, N=50, show=True, analyze=False)
print(stats)

defense.close()
attack.close()
normal.close()
training_hall.close()
env.close()
