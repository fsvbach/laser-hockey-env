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


#########################################################################################################
# GAMEPLAY


# env = h_env.HockeyEnv()
# load_weights = 'exp4'
# #load_weights = 'training_hall_1'

# td3 = TD3(pretrained='stronger')

# strong_basic_opponent = h_env.BasicOpponent(weak=False)
# weak_basic_opponent = h_env.BasicOpponent(weak=True) 



# q_agent = agent.DQNAgent(env.observation_space, env.discrete_action_space,
#                         convert_func =  env.discrete_to_continous_action,
#                         pretrained   = f'DQN/weights/{load_weights}')


# stats = gameplay(env, q_agent, player2=weak_basic_opponent, N=100, show=False, analyze=False)
# print("ties-wins-losses: ", stats)
# env.close()

#########################################################################################################
# TOURNAMENT


# env = h_env.HockeyEnv()
# load_weights = 'against_weak_45000'

# td3 = TD3(pretrained='superagent')

# strong_basic_opponent = h_env.BasicOpponent(weak=False)
# weak_basic_opponent = h_env.BasicOpponent(weak=True) 

# # ddpg = DDPGAgent(env,
# #                           actor_lr=1e-4,
# #                           critic_lr=1e-3,
# #                           update_rate=0.05,
# #                           discount=0.9, update_target_every=20,
# #                           pretrained='DDPG/weights/ddpg-normal-eps-noise-10000')

# # ddpg2 = DDPGAgent(env,
# #                           actor_lr=1e-4,
# #                           critic_lr=1e-3,
# #                           update_rate=0.05,
# #                           discount=0.9, update_target_every=20,
# #                           pretrained='DDPG/weights/ddpg-normal-eps-noise-basic-35000')


# # q_agent2 = agent.DQNAgent(env.observation_space, env.discrete_action_space,
# #                         convert_func =  env.discrete_to_continous_action,
# #                         pretrained   = 'DQN/weights/training_hall_1')


# q_agent = agent.DQNAgent(env.observation_space, env.discrete_action_space,
#                         convert_func =  env.discrete_to_continous_action,
#                         pretrained   = f'DQN/weights/{load_weights}')


# # agents = [weak_basic_opponent, strong_basic_opponent, q_agent, td3, ddpg2]
# # tournament = Tournament(env, agents)
# # tournament.run(50)
# # tournament.print_scores()
# # tournament.show_results()


# stats = gameplay(env, td3, player2=q_agent, N=20, show=True, analyze=False)
# print("ties-wins-losses: ", stats)

# env.close()
#########################################################################################################
# TOURNAMENT


# env = h_env.HockeyEnv()
# load_weights = 'training_hall:50000_omega=110_1_150_75_10_40000'

# td4 = TD3(pretrained='traininghall')
# td3 = TD3(pretrained='superagent')
# td5 = TD3(pretrained='stronger')
# strong_basic_opponent = h_env.BasicOpponent(weak=False)
# weak_basic_opponent = h_env.BasicOpponent(weak=True) 

# ddpg= DDPGAgent(env,
#                          actor_lr=1e-4,
#                          critic_lr=1e-3,
#                          update_rate=0.05,
#                          discount=0.9, update_target_every=20, pretrained="DDPG/weights/ddpg-normal-weak-10000")


# q_agent = agent.DQNAgent(env.observation_space, env.discrete_action_space,
#                         convert_func =  env.discrete_to_continous_action,
#                         pretrained   = f'DQN/weights/{load_weights}')

# agents = [weak_basic_opponent, strong_basic_opponent, td3, td4,td5]
# tournament = Tournament(env, agents)
# tournament.run(50)
# tournament.print_scores()
# tournament.show_results()





##########################################################################################################
# TRAINING HALL


load_weights = 'last_try_5000'
training_hall = TrainingHall2()

td3 = TD3(pretrained='superagent')

strong_basic_opponent = h_env.BasicOpponent(weak=False)

ddpg = DDPGAgent(pretrained="DDPG/weights/ddpg-checkpoint4")

q_agent = agent.DQNAgent(training_hall.observation_space, training_hall.discrete_action_space,
                        convert_func =  training_hall.discrete_to_continous_action,
                        pretrained   = f'DQN/weights/{load_weights}')


store_weights = "last_try"

agents = [strong_basic_opponent, td3, ddpg]
training_hall.register_opponents(agents)

losses, rewards = training.train(training_hall, q_agent, name=store_weights, show=False, max_episodes=50000)

stats = gameplay(training_hall, q_agent, N=50, show=True, analyze=False)
print("ties-wins-losses: ", stats)
training_hall.close()





#########################################################################################################
# GENETIC OPTIMIZATION

"""
GeneticOptimization(10, 25, 100, env=training_hall).run()
"""


#########################################################################################################
