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


attack = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)
defense = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_DEFENSE)

#########################################################################################################
# GAMEPLAY

# env = h_env.HockeyEnv()
# load_weights = 'training_hall:50000_omega=110_1_150_75_10_40000'

# td3 = TD3(pretrained='stronger')

# strong_basic_opponent = h_env.BasicOpponent(weak=False)
# weak_basic_opponent = h_env.BasicOpponent(weak=True) 

# ddpg = DDPGAgent(env,
#                           actor_lr=1e-4,
#                           critic_lr=1e-3,
#                           update_rate=0.05,
#                           discount=0.9, update_target_every=20,
#                           pretrained='DDPG/weights/ddpg-normal-eps-noise-10000')

# ddpg2 = DDPGAgent(env,
#                           actor_lr=1e-4,
#                           critic_lr=1e-3,
#                           update_rate=0.05,
#                           discount=0.9, update_target_every=20,
#                           pretrained='DDPG/weights/ddpg-normal-eps-noise-basic-35000')


# q_agent2 = agent.DQNAgent(env.observation_space, env.discrete_action_space,
#                         convert_func =  env.discrete_to_continous_action,
#                         pretrained   = 'DQN/weights/training_hall_1')


# q_agent = agent.DQNAgent(env.observation_space, env.discrete_action_space,
#                         convert_func =  env.discrete_to_continous_action,
#                         pretrained   = f'DQN/weights/{load_weights}')


# stats = gameplay(env, q_agent, player2=td3, N=50, show=True, analyze=True)
# print("ties-wins-losses: ", stats)

#########################################################################################################
# TOURNAMENT


# env = h_env.HockeyEnv()
# load_weights = 'training_hall:50000_omega=110_1_150_75_10_40000'



# td3 = TD3(pretrained='stronger')

# strong_basic_opponent = h_env.BasicOpponent(weak=False)
# weak_basic_opponent = h_env.BasicOpponent(weak=True) 

# ddpg = DDPGAgent(env,
#                           actor_lr=1e-4,
#                           critic_lr=1e-3,
#                           update_rate=0.05,
#                           discount=0.9, update_target_every=20,
#                           pretrained='DDPG/weights/ddpg-normal-eps-noise-10000')

# ddpg2 = DDPGAgent(env,
#                           actor_lr=1e-4,
#                           critic_lr=1e-3,
#                           update_rate=0.05,
#                           discount=0.9, update_target_every=20,
#                           pretrained='DDPG/weights/ddpg-normal-eps-noise-basic-35000')


# q_agent2 = agent.DQNAgent(env.observation_space, env.discrete_action_space,
#                         convert_func =  env.discrete_to_continous_action,
#                         pretrained   = 'DQN/weights/training_hall_1')


# q_agent = agent.DQNAgent(env.observation_space, env.discrete_action_space,
#                         convert_func =  env.discrete_to_continous_action,
#                         pretrained   = f'DQN/weights/{load_weights}')

# agents = [weak_basic_opponent, strong_basic_opponent, q_agent, td3, ddpg2]
# tournament = Tournament(env, agents)
# tournament.run(5)
# tournament.print_scores()
# tournament.show_results()






##########################################################################################################
# TRAINING HALL


#load_weights = 'training_hall:50000_omega=110_1_150_75_10'
load_weights = 'exp2_15000'
training_hall = TrainingHall2()

td3 = TD3(pretrained='stronger')

strong_basic_opponent = h_env.BasicOpponent(weak=False)
weak_basic_opponent = h_env.BasicOpponent(weak=True)
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


q_agent2 = agent.DQNAgent(training_hall.observation_space, training_hall.discrete_action_space,
                        convert_func =  training_hall.discrete_to_continous_action,
                        pretrained   = 'DQN/weights/training_hall_1')


q_agent = agent.DQNAgent(training_hall.observation_space, training_hall.discrete_action_space,
                        convert_func =  training_hall.discrete_to_continous_action,
                        pretrained   = f'DQN/weights/{load_weights}')


# weak basic opponent is in training per default
training_hall.register_opponents([strong_basic_opponent])

#store_weights = f'training_hall:50000_omega=1{q_agent._config["winner"]}_{q_agent._config["positioning"]}_{q_agent._config["distance_puck"]}_{q_agent._config["puck_direction"]}_{q_agent._config["touch_puck"]}'
store_weights = "exp3"

#losses, rewards = training.train(normal, q_agent, player2=basic_opponent, name=store_weights, show=False, max_episodes=10000)
losses, rewards = training.train(training_hall, q_agent, name=store_weights, show=False, max_episodes=10000)

stats = gameplay(training_hall, q_agent, N=10, show=True, analyze=False)
print("ties-wins-losses: ", stats)
training_hall.close()





#########################################################################################################
# GENETIC OPTIMIZATION

"""
GeneticOptimization(10, 25, 100, env=training_hall).run()
"""


#########################################################################################################
defense.close()
attack.close()
