import gym
import numpy as np
import torch
import memory
from ddpg_agent import DDPGAgent
from ou_noise import OUNoise
import matplotlib.pyplot as plt


def main():

    env_name = 'Pendulum-v0'
    env = gym.make(env_name)

    print("Actions ", env.action_space.shape[0])
    print("States ", env.observation_space.shape[0])

    ac_space = env.action_space
    o_space = env.observation_space

    ddpg_agent = DDPGAgent(o_space, ac_space, discount=0.99, batch_size=128, update_target_every=20, actor_lr=0.0001, critic_lr=0.001,
                      actor_update_rate=0.001, critic_update_rate=0.001, buffer_size=int(1e5))

    ob = env.reset()
    noise = OUNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))
    #ddpg_agent.actor.predict(ob)
    stats = []
    losses = []     

    max_episodes=800
    max_steps=200
    fps=50
    #mode="random"
    show=False

    #### Episode loop
    ###
    ##
    for i in range(max_episodes):
        #print("Starting a new episode")
        total_reward = 0
        noise.reset()
        ob = env.reset()
        for t in range(max_steps):
            done = False
        
            # ob is (2, 1) but we need (batch, features) = (1, 2)
            a = ddpg_agent.actor.get_action(ob.T).detach().numpy().flatten() + noise()

            (ob_new, reward, done, _info) = env.step(a)
            total_reward += reward
            ddpg_agent.store_transition((ob, torch.from_numpy(a), reward, ob_new, done))            
            ob=ob_new
            if show:
                time.sleep(1.0/fps)
                env.render(mode='human')        
            if done: break    
        if ddpg_agent.buffer.size > ddpg_agent._config["batch_size"]:
            losses.extend(ddpg_agent.train(32))
        stats.append([i,total_reward,t+1])    
    
        if ((i-1)%20==0):
            print("{}: Done after {} steps. Reward: {}".format(i, t+1, total_reward))
            #print("Loss: ", losses[-1])
    
    plt.plot(losses)


if __name__ == "__main__":
    main()