import time
import numpy as np
import matplotlib.pyplot as plt

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def train(env, q_agent, player2, max_episodes=200, max_steps=300, show=False, name='test'):
    
    stats = []
    losses = []
    fps=50
    
    for i in range(max_episodes):
        if i % (max_episodes/10) == 0 and i != 0: 
            q_agent.reduce_exploration(0.1)
            q_agent.save_weights(f'DQN/weights/{name}_{i}')
            plt.plot(running_mean(losses,64))
            plt.show()
            print("current buffer size: ", q_agent.buffer.size)
        # print("Starting a new episode")    
        total_reward = 0
        obs2 = ob = env.reset(mode=i%3)
        for t in range(max_steps):
            done = False        
            a1 = q_agent.act(ob)
            a2 = [0,0.,0,0] 
            if env.mode == 0:
                a2 = player2.act(obs2)
            (ob_new, reward, done, _info) = env.step(np.hstack([a1,a2]))
            total_reward+= reward
            if show:
                time.sleep(1.0/fps)
                env.render(mode='human')        
            q_agent.store_transition([ob, reward, ob_new, done])  
            if done: 
                break    
            ob=ob_new        
            obs2 = env.obs_agent_two()
            
        # print('buffer_size',q_agent.buffer.size)
        losses.extend(q_agent.train(32))
        stats.append([i,total_reward,t+1])    
        
        if ((i-1)%20==0):
            print("Episode {}: Done after {} steps. Reward: {}".format(i, t+1, total_reward))
        
        # if ((i-1)%10000==0):
        #     q_agent.save_weights(f'DQN/weights/{name}_{i}')
            
    q_agent.save_weights(f'DQN/weights/{name}')
    rewards =  [r[1] for r in stats]
    
    plt.plot(running_mean(losses,64))
    plt.savefig(f'Plots/{name}_losses')
    plt.show()
    plt.close()
    
    plt.plot(running_mean(rewards,200))
    plt.savefig(f'Plots/{name}_rewards')
    plt.show()
    plt.close()

    return losses, rewards
