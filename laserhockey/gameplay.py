#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 13:33:27 2021

@author: fsvbach
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from laserhockey.TrainingHall import TrainingHall
import laserhockey.hockey_env as h_env


def gameplay(env, player1, player2=False, N=1, show=False, analyze=False):
    win_stats = np.zeros((N,3))
    max_len = 500
    fps = 100
    training_hall = isinstance(env, TrainingHall)
    for n in range(N):
        obs = env.reset()
        obs_agent2 = env.obs_agent_two()
        total_reward=0
        for _ in range(max_len):
            a1 = player1.act(obs, eps=0) 
            if training_hall: 
                (ob_new, reward, done, _info) = env.step(a1)
            else:
                a2 = [0,0.,0,0] 
                if player2:
                    a2 = player2.act(obs_agent2)
                obs, r, done, _info = env.step(np.hstack([a1,a2]))    
                obs_agent2 = env.obs_agent_two()
            if show:
                time.sleep(1/fps)
                if analyze:
                    time.sleep(50/fps)
                    print('winner:',_info['winner'])
                    print("punishment_positioning: ", _info["punishment_positioning"])
                    print("punishment_distance_puck: ", _info["punishment_distance_puck"])
                    print("reward_puck_direction: ", _info["reward_puck_direction"])
                    print("reward_touch_puck: ", _info["reward_touch_puck"])
                    print('...')
                    print(total_reward)
                env.render()
            if done: break
        win_stats[n, env._get_info()['winner']] = 1
    return np.sum(win_stats, axis = 0)



"ties-wins-losses"
class Tournament: 
    
    def __init__(self, env, agents): 
        self.agents = agents
        self.env = env
        # results contains the results for the basic metric and the football metric in two matrices
        n = len(agents)
        self.results = np.zeros((2, n, n))
        
    def run(self, rounds=100): 
        for i, player1 in enumerate(self.agents): 
            for j, player2 in enumerate(self.agents): 
                if i==j: 
                    print("same agent reached")
                    continue
                stats = gameplay(self.env, player1, player2, rounds, show=True)
                # results based on the basic metric
                self.results[0][i][j] = stats[1] - stats[2]
                self.results[0][j][i] = stats[2] - stats[1]
                # results based on the soccer metric
                self.results[1][i][j] = stats[0] + 3*stats[1]
                self.results[1][j][i] = stats[0] + 3*stats[2]
    
    def show_results(self): 
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].set_title("basic metric")
        axs[1].set_title("soccer metric")
        axs[0].imshow(self.results[0], cmap='coolwarm', interpolation='nearest')
        axs[1].imshow(self.results[1], cmap='coolwarm', interpolation='nearest')
        plt.colormap()
        plt.show()


        














