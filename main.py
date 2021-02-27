from DQN import agent
from gym import spaces
import time
import laserhockey.hockey_env as h_env

env = h_env.HockeyEnv()
ac_space = spaces.discrete.Discrete(2)
o_space = env.observation_space

q_agent = agent.DQNAgent(o_space, ac_space, discount=0.95, eps=0.2, update_rule=20)

stats = []
losses = []

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