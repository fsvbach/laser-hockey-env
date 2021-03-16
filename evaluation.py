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

td5 = TD3(pretrained='continuation')
td4 = TD3(pretrained='superagent')
td3 = TD3(pretrained='overfit')
td2 = TD3(pretrained='traininghall')
td1 = TD3(pretrained='td3')


strong_basic_opponent = h_env.BasicOpponent(weak=False)
weak_basic_opponent = h_env.BasicOpponent(weak=True) 

ddpg = DDPGAgent(pretrained="DDPG/weights/ddpg-checkpoint4")

q_agent = agent.DQNAgent(env.observation_space, env.discrete_action_space,
                        convert_func =  env.discrete_to_continous_action,
                        pretrained   = 'DQN/weights/alg2')


agents = [weak_basic_opponent, strong_basic_opponent, td5,ddpg,q_agent]
tournament = Tournament(env, agents)
tournament.run(100)
tournament.print_scores()
tournament.show_results()


