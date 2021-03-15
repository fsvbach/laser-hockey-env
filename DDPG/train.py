import time
import numpy as np
import torch
from .ddpg_agent import OUNoise
import sys


def train(env, agent, player2=False, max_episodes=200, max_steps=300, show=False, name='test', reward_weights=[3, 2, 1, 1, 2]):
    """
    Perform training with the agent for given number of episodes and saves it under given name.
        Args:
            env: Environment
            agent: Agent to train
            player2: Other agent
            max_episodes: Number of episodes to run
            max_steps: Number of steps per episode
            show: Visual or not
            name: Name to save model under
            reward_weights: Weighing factors for reward features
    
        Returns:
            losses and stats for the model

    """

    ob = env.reset()
    noise = OUNoise(action_dim=env.num_actions, 
                        action_low=env.action_space.low[:4],
                        action_high=env.action_space.high[:4],
                        max_sigma=0.3, min_sigma=0.2)

    stats = []
    losses = []
    rewards = []
    fps=50
    show = False
    eps = 1
    eps_cnt = 1

    for i in range(max_episodes):
        noise.reset()
        ob = env.reset()
        ob2 = env.reset()
        total_reward = 0

        if (i > eps_cnt * max_episodes / 10):
            eps -= 0.15
            if (eps < 0):
                eps = 0
            eps_cnt += 1

        for step in range(max_steps):
            done = False
            act = agent.act(ob, eps=eps)
            act = noise.get_action(act, step)
            act2 = [0,0.,0,0]
            if player2:
                act2 = player2.act(ob2)

            (ob_new, reward, done, _info) = env.step(np.hstack([act,act2]))

            reward = reward_weights[0] * _info["winner"] + reward_weights[1] * _info['punishment_distance_puck'] + reward_weights[2] * _info["reward_touch_puck"] 
            + reward_weights[3] * _info["reward_puck_direction"] + reward_weights[4] * _info['punishment_positioning']

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

    agent.save_weights(f'DDPG/weights/{name}')

    return losses, [r[1] for r in stats]
