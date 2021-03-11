#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:16:26 2021

@author: johannes
"""

from DQN import agent, training
from gym.spaces.discrete import Discrete
import laserhockey.hockey_env as h_env
from laserhockey.gameplay import gameplay
from DDPG.ddpg_agent import DDPGAgent
from TD3.agent import TD3
import TrainingHall

env = h_env.HockeyEnv()
attack = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
defense = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)
normal = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
training_hall = TrainingHall()

load_weights = 'symmetric_weights_strong_opponent'
store_weights = 'symmetric_weights_strong_opponent2'

td3 = TD3(18, 4, 1.0, env)
td3.load(filename='stronger')
basic_opponent = h_env.BasicOpponent(weak=False)
ddpg_agent = DDPGAgent(env,
                         actor_lr=1e-4,
                         critic_lr=1e-3,
                         update_rate=0.05,
                         discount=0.9, update_target_every=20,
                         pretrained='DDPG/weights/ddpg-normal-eps-noise-10000')

q_agent = agent.DQNAgent(env.observation_space, env.discrete_action_space,
                        convert_func =  env.discrete_to_continous_action,
                        pretrained   = f'DQN/weights/{load_weights}')

training_hall.register_opponents([ddpg_agent, td3, basic_opponent])

losses, rewards = training.train(normal, q_agent, player2=basic_opponent, name=store_weights, show=False, max_episodes=10000)


stats = gameplay(training_hall, q_agent, player2=basic_opponent, N=50, show=True, analyze=False)
print(stats)

defense.close()
attack.close()
normal.close()
training_hall.close()
env.close()
