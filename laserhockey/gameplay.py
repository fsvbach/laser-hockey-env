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
import pandas as pd


def gameplay(env, player1, player2=False, N=1, show=False, analyze=False):
    win_stats = np.zeros((N,3),dtype=np.int)
    max_len = 500
    fps = 100
    for n in range(N):
        obs = env.reset()
        if not player2: 
            print("playing against: ", env.opponent.name())
        total_reward=0
        for _ in range(max_len):
            a1 = player1.act(obs, eps=0) 
            if player2: 
                obs_agent2 = env.obs_agent_two()
                a2 = player2.act(obs_agent2)
                obs, reward, done, _info = env.step(np.hstack([a1,a2]))    
            else: 
                obs, reward, done, _info = env.step(a1)
            total_reward += reward + sum(list(_info.values()))
            if show:
                time.sleep(1/fps)
                if analyze:
                    time.sleep(50/fps)
                    print("reward_positioning: ", _info["punishment_positioning"])
                    print("punishment_distance_puck: ", _info["punishment_distance_puck"])
                    print("puck_direction: ", _info["reward_puck_direction"])
                    print("reward_touch_puck: ", _info["reward_touch_puck"])
                    print('TOTAL: ' ,total_reward)
                    print('----------------------')
                env.render()
            if done: break
        win_stats[n, env._get_info()['winner']] = 1
        input()
    return np.sum(win_stats, axis = 0)



class Tournament: 
    
    def __init__(self, env, agents): 
        self.agents = agents
        self.env = env
        n = len(agents)
        self.results_basic = np.zeros((n, n))
        self.results_soccer = np.zeros((n, n))
        self.total_scores = np.zeros((n,3), dtype=np.int)
        
    def run(self, rounds=100): 
        show = False
        for i, player1 in enumerate(self.agents): 
            for j, player2 in enumerate(self.agents): 
                if i == j:
                    # results based on the basic metric
                    self.results_basic[i][j] = None
                    # results based on the soccer metric
                    self.results_soccer[i][j] = None
                    continue
                # show = player1.name() =="DQN"
                stats = gameplay(self.env, player1, player2, rounds, show=show)              
                
                # results based on the basic metric
                self.results_basic[i][j] += (stats[1] - stats[2])/rounds/2
                self.results_basic[j][i] += (stats[2] - stats[1])/rounds/2
                # results based on the soccer metric
                self.results_soccer[i][j] += (stats[0] + 3*stats[1])/rounds /2
                self.results_soccer[j][i] += (stats[0] + 3*stats[2])/rounds /2
                
                #total scores summing
                self.total_scores[i] += stats[[1,0,2]]
                self.total_scores[j] += stats[[2,0,1]]
                
        self.compute_scores()

    def compute_scores(self):
        self.soccer_scores = np.round(np.nanmean(self.results_soccer, axis=1),3)
        self.basic_scores =np.round(np.nanmean(self.results_basic, axis=1),3)
        
    def print_scores(self):
        print("\nTotal Ties-Wins-Losses:")
        table = pd.DataFrame(self.total_scores, 
                             index = [a.name() for a in self.agents],
                             columns=['Wins','Ties','Losses'])
        table['Soccer Score'] = self.soccer_scores
        table['Basic Score'] = self.basic_scores
        table = table.sort_values('Soccer Score', ascending=False)
        table.to_csv('Plots/tournament_scores.csv')
        print(table)

    
    
    def show_results(self): 
        fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(20,12))
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
        plt.savefig('Plots/tournament_results.svg')
        plt.show()









