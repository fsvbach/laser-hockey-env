#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 13:33:27 2021

@author: fsvbach
"""

import numpy as np
import time

def gameplay(env, player1, player2=False, N=1, show=False, analyze=False):
    win_stats = np.zeros((N,3))
    max_len = 500
    fps = 100

    for n in range(N):
        obs = env.reset()
        obs_agent2 = env.obs_agent_two()
        for _ in range(max_len):
            a1 = player1.act(obs, eps=0) 
            a2 = [0,0.,0,0] 
            if player2:
                a2 = player2.act(obs_agent2)
            obs, r, d, _info = env.step(np.hstack([a1,a2]))    
            obs_agent2 = env.obs_agent_two()
            if show:
                time.sleep(1/fps)
                if analyze:
                    time.sleep(25/fps)
                    print("punishment_positioning: ", _info["punishment_positioning"])
                    print("punishment_distance_puck: ",100* _info["punishment_distance_puck"])
                    print("reward_puck_direction: ", 100*_info["reward_puck_direction"])
                    print("reward_touch_puck: ", _info["reward_touch_puck"])
                env.render()
            if d: break
        
        win_stats[n, env._get_info()['winner']] = 1
    return np.sum(win_stats, axis = 0)

