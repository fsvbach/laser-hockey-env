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
from laserhockey.TrainingHall import TrainingHall2
from laserhockey.gameplay import Tournament


env = h_env.HockeyEnv()
load_weights = 'alg2'

td4 = TD3(pretrained='best_avg')
td3 = TD3(pretrained='superagent')
td5 = TD3(pretrained='lasttry')


strong_basic_opponent = h_env.BasicOpponent(weak=False)
weak_basic_opponent = h_env.BasicOpponent(weak=True) 

ddpg = DDPGAgent(pretrained="DDPG/weights/checkpoint4")

q_agent = agent.DQNAgent(env.observation_space, env.discrete_action_space,
                        convert_func =  env.discrete_to_continous_action,
                        pretrained   = f'DQN/weights/{load_weights}')


agents = [weak_basic_opponent, strong_basic_opponent, td3, ddpg, q_agent]
tournament = Tournament(env, agents)
tournament.run(10)
tournament.print_scores()
tournament.show_results()



