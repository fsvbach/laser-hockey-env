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
        self.results = np.zeros((n,n),  dtype=(float,3))
        self.results_basic = np.zeros((n, n))
        self.results_soccer = np.zeros((n, n))
        self.total_scores = np.zeros((n,3))
        
    def run(self, rounds=100): 
        show = False
        for i, player1 in enumerate(self.agents): 
            for j, player2 in enumerate(self.agents): 
                if i == j:
                    # results based on the basic metric
                    self.results_basic[i][j] = None
                    self.results_basic[j][i] = None
                    # results based on the soccer metric
                    self.results_soccer[i][j] = None
                    self.results_soccer[j][i] = None
                    continue
                show = player1.name() =="DQN"
                stats = gameplay(self.env, player1, player2, rounds, show=show)
                self.results[i][j] = (stats[0], stats[1], stats[2])
                self.results[j][i] = (stats[0], stats[2], stats[1])
                
                
                #compute sum of all all losses ties and wins for every player
                for k in range(3):
                    self.total_scores[i][k] += stats[k]
                    self.total_scores[j][k] += stats[k]
                
                
                # results based on the basic metric
                self.results_basic[i][j] = (stats[1] - stats[2])/rounds
                self.results_basic[j][i] = (stats[2] - stats[1])/rounds
                # results based on the soccer metric
                self.results_soccer[i][j] = (stats[0] + 3*stats[1])/rounds 
                self.results_soccer[j][i] = (stats[0] + 3*stats[2])/rounds 
        self.compute_scores()

    def compute_scores(self):
        self.soccer_scores = np.nanmean(self.results_soccer, axis=1)
        self.basic_scores = np.nanmean(self.results_basic, axis=1)
        
    def print_scores(self):
        print("\nTotal Ties-Wins-Losses:")
        for i,a in enumerate(self.agents): 
            print (a.name(), ": ", self.total_scores[i])
        print("\nSoccer Scores: ", self.soccer_scores)
        print("\nBasic Scores: ", self.basic_scores)
    
    
    def show_results(self): 
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        axs[0].set_title("basic metric")
        axs[1].set_title("soccer metric")
        
        labels = [""] + [a.name() for a in self.agents]
        print(labels)
        axs[0].set_xticklabels(labels, minor=False)
        axs[0].set_yticklabels(labels, minor=False)
        axs[1].set_xticklabels(labels, minor=False)
        axs[1].set_yticklabels(labels, minor=False)
        
        basic = axs[0].imshow(self.results_basic, cmap='RdYlGn', interpolation='nearest')
        soccer = axs[1].imshow(self.results_soccer, cmap='Greens', interpolation='nearest')
        plt.colorbar(mappable=basic, ax=axs[0])
        plt.colorbar(mappable=soccer, ax=axs[1])
        plt.show()









