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
from laserhockey.TrainingHall2 import TrainingHall2
from laserhockey.gameplay import Tournament


env = h_env.HockeyEnv()
load_weights = 'training_hall:50000_omega=110_1_150_75_10'

td4 = TD3(pretrained='best_avg')
td3 = TD3(pretrained='superagent')
td5 = TD3(pretrained='lasttry')


strong_basic_opponent = h_env.BasicOpponent(weak=False)
weak_basic_opponent = h_env.BasicOpponent(weak=True) 

ddpg= DDPGAgent(env,
                          actor_lr=1e-4,
                          critic_lr=1e-3,
                          update_rate=0.05,
                          discount=0.9, update_target_every=20, pretrained="DDPG/weights/ddpg-normal-weak-35000")

q_agent = agent.DQNAgent(env.observation_space, env.discrete_action_space,
                        convert_func =  env.discrete_to_continous_action,
                        pretrained   = f'DQN/weights/{load_weights}')


agents = [ strong_basic_opponent, td3, td4,td5,ddpg,q_agent]
tournament = Tournament(env, agents)
tournament.run(10)
tournament.print_scores()
tournament.show_results()



