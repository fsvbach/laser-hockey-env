#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:24:40 2021

@author: johannes
"""

import numpy as np
from .agent import DQNAgent
from laserhockey.gameplay import gameplay
import laserhockey.hockey_env as h_env
from laserhockey.TrainingHall import TrainingHall
from .training import train_gen_opt
from DDPG.ddpg_agent import DDPGAgent

# possible ranges for hyperparameters [10,1,200,100,10] currently best
config_ranges = {
            "eps": [0.5, 0.75, 1],                  
            "discount": [0.9, 0.95, 0.99],
            "buffer_size": [int(1e4)],
            "batch_size": [128],
            "learning_rate": [0.001, 0.0005, 0.0001], 
            "update_rule": [10,20,30,40],
            "multistep": [2,4,8],
            "omega": [0.5, 1],
            "winner": np.arange(2, 20, 2),
            "positioning": np.arange(0.2, 2, 0.2),
            "distance_puck": np.arange(30, 300, 30),
            "puck_direction": np.arange(30, 300, 30),
            "touch_puck": np.arange(3, 30, 3)
        }



class GeneticOptimization: 
    
    def __init__(self, generations, population_size, train_episodes, env=h_env.HockeyEnv(), survival_rate=0.2):
        self.generations = generations
        self.env = env
        self.population_size = population_size
        self.train_episodes = train_episodes
        self.population = self.construct_population()
        self.survival_rate = survival_rate
        self.store_weights = f"gen={self.generations}_pop={self.population_size}_ep={self.train_episodes}"
        
    def construct_population(self, parents=None):
        if parents:
            num_children = int(self.population_size/len(parents))
            children = parents.copy()
            for p in parents: 
                parent_genes = p._config
                for child in range(num_children - 1): 
                    child_genes = self.mutate_genes(parent_genes)
                    agent = DQNAgent(self.env.observation_space, self.env.discrete_action_space,
                        convert_func =  self.env.discrete_to_continous_action, userconfig=child_genes)
                    agent.buffer = p.buffer.copy()
                    agent.Q.load_state_dict(p.Q.state_dict())
                    agent.T.load_state_dict(p.T.state_dict())
                    children.append(agent)
            return children
                    
        else:
          population = []
          for i in range(self.population_size): 
            genes = self.create_genes()
            agent = DQNAgent(self.env.observation_space, self.env.discrete_action_space,
                        convert_func =  self.env.discrete_to_continous_action, userconfig=genes)
            population.append(agent)
          return population
            
    
    def create_genes(self): 
        gene = dict()
        for key in config_ranges: 
            gene[key] = np.random.choice(config_ranges[key])
        return gene
            
    def mutate_genes(self, genes, p=0.1): 
        new_genes = dict()
        for key in genes: 
            if np.random.random() < p:
                new_genes[key] = np.random.choice(config_ranges[key])
            else: 
                new_genes[key] = genes[key]
        return new_genes
            
    def evaluate_population(self):
        scores = [-self.population_size] * self.population_size
        print("starting evaluation")
        for i,agent in enumerate(self.population): 
            print(f"evaluating {i+1}.th agent")
            stats = gameplay(self.env, agent, N=20)
            scores[i] = stats[1] - stats[2]
        combined = list(zip(scores, self.population))
        combined.sort(key=lambda tup: tup[0], reverse=True)
        print("top 20 % scores: ", [y for y,_ in combined][:int(self.population_size*self.survival_rate)])
        print("top 20 % configs", [x._config for _,x in combined][:int(self.population_size*self.survival_rate)])
        return [x for _,x in combined][:int(self.population_size*self.survival_rate)]
    
    def train_population(self): 
        for i, agent in enumerate(self.population): 
            print(f"training {i+1}.th agent")
            train_gen_opt(self.env, agent, player2=False, max_episodes=self.train_episodes)

            
    def run(self): 
        for g in range(self.generations): 
            print(f"Training generation {g+1}:")
            self.train_population()
            parents = self.evaluate_population()
            self.population = self.construct_population(parents)
        best_agents = self.evaluate_population()
        for i, agent in enumerate(best_agents): 
            agent.save_weights(f'DQN/weights/{self.store_weights}_agent={i+1}')
            stats = gameplay(self.env, agent, N=2, show=True)
            print("config, win_rates: ", agent._config, stats)
            
            
            
        










