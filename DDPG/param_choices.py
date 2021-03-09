params1 = DDPGAgent(env.observation_space, 
                         env.action_space,
                         actor_lr=1e-5,
                         critic_lr=1e-4,
                         update_rate=0.01,
                         discount=0.95)             

params2 = DDPGAgent(env.observation_space, 
                         env.action_space,
                         actor_lr=1e-4,
                         critic_lr=1e-4,
                         update_rate=0.05,
                         discount=0.95)           