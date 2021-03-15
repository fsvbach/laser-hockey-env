#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 08:05:04 2021

@author: johannes
"""
import sys
sys.path.append('/home/johannes/Uni/ReinforcementLearning/project/DQN')

from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client



from DQN import DQNAgent
from laserhockey.hockey_env import HockeyEnv


load_weights = "DQN/weights/training_hall_1"
env = HockeyEnv()


class RemoteDQNAgent(DQNAgent):

    def __init__(self):
        
        DQNAgent.__init__(self, env.observation_space, env.discrete_action_space,
                        convert_func =  env.discrete_to_continous_action,
                        pretrained   = f'DQN/weights/{load_weights}')
        RemoteControllerInterface.__init__(self, identifier='StillTrying_DQN')

    def remote_act(self, obs):
        return self.act(obs,epsilon=0)
        

if __name__ == '__main__':
    controller = RemoteDQNAgent()

    # Play n (None for an infinite amount) games and quit
    client = Client(username='Johannes_Schulz_StillTrying', # Testuser
                    password='Rba*9UxK',
                    controller=controller, 
                    output_path='/ALRL2020/games', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=None)

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queuing
    # client = Client(username='user0', 
    #                 password='1234',
    #                 controller=controller, 
    #                 output_path='/tmp/ALRL2020/client/user0',
    #                )
