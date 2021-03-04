import time
import numpy as np

def train(env, q_agent, player2=False, max_episodes=200, max_steps=300, show=False, name='test'):
    
    stats = []
    losses = []
    fps=50
    
    for i in range(max_episodes):
        # print("Starting a new episode")    
        total_reward = 0
        obs2 = ob = env.reset()
        for t in range(max_steps):
            done = False        
            a1 = q_agent.act(ob)
            a2 = [0,0.,0,0] 
            if player2:
                a2 = player2.act(obs2)
            (ob_new, reward, done, _info) = env.step(np.hstack([env.discrete_to_continous_action(a1),a2]))
            total_reward+= reward
            q_agent.store_transition((ob, a1, reward, ob_new, done))            
            ob=ob_new        
            obs2 = env.obs_agent_two()
            if show:
                time.sleep(1.0/fps)
                env.render(mode='human')        
            if done: break    
        # print('buffer_size',q_agent.buffer.size)
        losses.extend(q_agent.train(32))
        stats.append([i,total_reward,t+1])    
        
        if ((i-1)%20==0):
            print("Episode {}: Done after {} steps. Reward: {}".format(i, t+1, total_reward))
            
    
    q_agent.save_weights(f'DQN/weights/{name}')

    return losses, [r[1] for r in stats]
