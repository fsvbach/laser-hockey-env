import time
import numpy as np
import torch
from .ddpg_agent import OUNoise
import sys

def train(env, agent, player2=False, max_episodes=200, max_steps=200, show=False, name='test', reward_weights=[3, 2, 1, 1, 2]):

    ob = env.reset()
    
    noise = OUNoise(action_dim=env.num_actions, 
                        action_low=env.action_space.low[:4], 
                        action_high=env.action_space.high[:4],
                        max_sigma=0.3, min_sigma=0.2)

    stats = []
    losses = []
    rewards = []
    avg_rewards = []
    fps=50
    show = False
    eps = 1
    eps_cnt = 1

    for i in range(max_episodes):
        noise.reset()
        ob = env.reset()
        ob2 = env.reset()
        episode_losses = 0
        total_reward = 0

        if (i > eps_cnt * max_episodes / 10):
            eps -= 0.2
            if (eps < 0):
                eps = 0
            eps_cnt += 1

        for step in range(max_steps):
            done = False
            act = agent.act(ob, eps=eps)
            act = noise.get_action(act, step)
            act2 = [0,0.,0,0]

            if player2:
                # randomly select player
                #act2 = player2[np.random.randint(0,2)].act(ob2)
                act2 = player2.act(ob2)

            (ob_new, reward, done, _info) = env.step(np.hstack([act,act2]))
            # touch puck höher, muss trotzdem durch touch puck ausgleichbar sein
            # positioning bisschen höher
            # reward komplett rausnehmen? (winner höher skalieren)
            # reward puck direction raus
            # tore schießen höher als tor bekommen
            # reward rausgenommen
            #reward = reward + reward_weights[0] * _info["winner"] + reward_weights[1] * _info['punishment_distance_puck'] + reward_weights[2] * _info["reward_touch_puck"] + reward_weights[3] * _info["reward_puck_direction"]
            #+ reward_weights[4] * _info['punishment_positioning']
            reward = reward_weights[0] * _info["winner"] + reward_weights[1] * _info['punishment_distance_puck'] + reward_weights[2] * _info["reward_touch_puck"] + reward_weights[3] * _info["reward_puck_direction"]
            + reward_weights[4] * _info['punishment_positioning']

            total_reward += reward
            agent.store_transition((ob, act, reward, ob_new, done))      
            
            ob=ob_new
            ob2 = env.obs_agent_two()

            if show:
                time.sleep(1.0/fps) 
                env.render(mode='human')

            if done: break    

        losses.extend(agent.train(32))
        stats.append([i,total_reward,step+1])    
    
        if ((i-1)%20==0):
            print("{}: Done after {} steps. Reward: {}".format(i, step+1, total_reward))
    
        if (i % 5000) == 0:
            agent.save_weights(f'DDPG/weights/{name}_{i}')

    return losses, [r[1] for r in stats]


def alt_train(env, agent, player2=False, max_episodes=200, max_steps=200, show=False, name='test'):
    ob = env.reset()
    noise = OUNoise(env.action_space)
    rewards = []
    avg_rewards = []
    losses = []

    for episode in range(max_episodes):
        state = env.reset()
        noise.reset()
        episode_reward = 0
        episode_loss = 0
    
        for step in range(max_steps):
            
            done = False
            act = agent.act(ob)
            act = noise.get_action(act, step)
            act2 = [0,0.,0,0] 

            if player2:
                act2 = player2.act(obs2)

            (ob_new, reward, done, _info) = env.step(np.hstack([act,act2]))
            agent.store_transition((ob, act, reward, ob_new, done))    

            ob = ob_new
            episode_reward += reward
        
            if show:
                time.sleep(1.0/50)
                env.render(mode='human')  

            if done:
                sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-20:])))
                break
        
        losses.append(agent.train(32))
        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-20:]))

    return rewards, avg_rewards

