from DQN import agent
from gym.spaces.discrete import Discrete
import time
import numpy as np
import laserhockey.hockey_env as h_env

env = h_env.HockeyEnv()
ac_space = Discrete(8)
o_space = env.observation_space

q_agent = agent.DQNAgent(o_space, ac_space, discount=0.95, eps=0.2, update_rule=20)
player2 = h_env.BasicOpponent()

stats = []
losses = []

max_episodes=600
max_steps=500
fps=50
#mode="random"
show=False
for i in range(max_episodes):
    print("Starting a new episode")    
    total_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        done = False        
        a1 = q_agent.act(ob)
        a1_cont = env.discrete_to_continous_action(a1)
        a2 = player2.act(ob)
        (ob_new, reward, done, _info) = env.step(np.hstack([a1_cont,a2]))
        total_reward+= reward
        q_agent.store_transition((ob, a1, reward, ob_new, done))            
        ob=ob_new        
        if show:
            time.sleep(1.0/fps)
            env.render(mode='human')        
        if done: break    
    print('buffer_size',q_agent.buffer.size)
    losses.extend(q_agent.train(32))
    stats.append([i,total_reward,t+1])    
    
    if ((i-1)%20==0):
        print("{}: Done after {} steps. Reward: {}".format(i, t+1, total_reward))