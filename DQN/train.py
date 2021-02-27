
import numpy as np
import itertools
import time
import torch
import pylab as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import memory as mem   
from feedforward import Feedforward
#import MountainCarInformativeReward
from custompendulumenv import CustomPendulumEnvDiscrete


q_agent = agent.DQNAgent(o_space, ac_space, discount=0.95, eps=0.2, update_rule=20)

max_episodes=600
max_steps=500
fps=50
#mode="random"
show=False
for i in range(max_episodes):
    # print("Starting a new episode")    
    total_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        done = False        
        a = q_agent.act(ob)
        (ob_new, reward, done, _info) = env.step(a)
        total_reward+= reward
        q_agent.store_transition((ob, a, reward, ob_new, done))            
        ob=ob_new        
        if show:
            time.sleep(1.0/fps)
            env.render(mode='human')        
        if done: break    
    #print('buffer_size',q_agent.buffer.size)
    losses.extend(q_agent.train(32))
    stats.append([i,total_reward,t+1])    
    
    if ((i-1)%20==0):
        print("{}: Done after {} steps. Reward: {}".format(i, t+1, total_reward))


stats_np = np.asarray(stats)

def plot_stats(stats, N, name):
    title= f'{name} k={q_agent._config["update_rule"]}'
    plt.figure()
    plt.plot(running_mean(stats[:,1], N), label='rewards')
    plt.plot(running_mean(stats[:,2], N), label='steps')
    plt.title(title)
    plt.savefig(f"{title}.pdf", bbox_inches="tight")
    plt.legend()
    
plot_stats(stats_np, 20, 'cartpole train ')
q_agent