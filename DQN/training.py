import time
import numpy as np
import matplotlib.pyplot as plt
from laserhockey.gameplay import gameplay
import laserhockey.hockey_env as h_env

punishment_positioning = 1
punishment_distance_puck = 200 
reward_puck_direction = 100
reward_touch_puck = 10
reward_winner = 10
fps=2


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def train(env, q_agent, player2=False, max_episodes=200, max_steps=300, name='test', show=False):
    
    stats = []
    losses = []
    
    for i in range(max_episodes):
        if i % (max_episodes/10) == 0: 
            if i != 0:
                q_agent.reduce_exploration(0.1* q_agent._config['eps'])
                #q_agent._config['discount'] += 0.02
                #q_agent.save_weights(f'DQN/weights/{name}_{i}')
            
            # create plots from losses and rewards till now
            rewards =  [r[1] for r in stats]
            fig, axs = plt.subplots(2, 1, constrained_layout=True)
            axs[0].set_title("rewards")
            axs[1].set_title("losses")
            axs[0].plot(running_mean(rewards, 64))
            axs[1].plot(running_mean(losses, 64))
            plt.show()
            
            normal = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
            if not player2: 
                player2 = env.opponent
            winrate = gameplay(normal, q_agent, player2, N=10, show=True, analyze=False)
            print("ties-wins-losses: ", winrate)
            normal.close()
            
        total_reward = 0
        ob = env.reset()
        for t in range(max_steps):
            a1 = q_agent.act(ob)
            (ob_new, reward, done, _info) = (None, None, None, None)
            if player2:     
                ob2 = env.obs_agent_two()
                a2 = [0,0.,0,0] 
                if env.mode == 0:
                    a2 = player2.act(ob2)
                (ob_new, reward, done, _info) = env.step(np.hstack([a1,a2]))
            else:
                #environment handles opponents action
                (ob_new, reward, done, _info) = env.step(a1)
                
            reward *= reward_winner
            reward += punishment_positioning *_info["punishment_positioning"] + punishment_distance_puck*_info["punishment_distance_puck"] + reward_puck_direction*_info["reward_puck_direction"] + reward_touch_puck*_info["reward_touch_puck"]
            total_reward+= reward       
            q_agent.store_transition([ob, reward, ob_new, done])  
            if show:
                time.sleep(1.0/fps)
            if done: 
                break    
            ob=ob_new  
            
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
    plt.savefig(f'DQN/plots/{name}_losses')
    plt.show()
    plt.close()
    
    plt.plot(running_mean(rewards,200))
    plt.savefig(f'DQN/plots/{name}_rewards')
    plt.show()
    plt.close()

    return losses, rewards
