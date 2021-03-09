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
                         
params3 = DDPGAgent(env.observation_space, 
                         env.action_space,
                         actor_lr=1e-4,
                         critic_lr=1e-3,
                         update_rate=0.01,
                         discount=0.98)  

params4 = DDPGAgent(env.observation_space, 
                         env.action_space,
                         actor_lr=1e-5,
                         critic_lr=1e-3,
                         update_rate=0.01,
                         discount=0.9,
                         update_target_every=5)  

params5 = DDPGAgent(env.observation_space, 
                         env.action_space,
                         actor_lr=1e-4,
                         critic_lr=1e-3,
                         update_rate=0.01,
                         discount=0.9, update_target_every=20)

params6 = DDPGAgent(env.observation_space, 
                         env.action_space,
                         actor_lr=1e-4,
                         critic_lr=1e-4,
                         update_rate=0.05,
                         discount=0.9, update_target_every=20)

params7 = DDPGAgent(env,
                         actor_lr=1e-4,
                         critic_lr=1e-3,
                         update_rate=0.05,
                         discount=0.9, update_target_every=20)

param_grid = [[1e-5, 1e-4, 0.01, 0.95], [1e-5, 1e-4, 0.01, 0.99], [1e-5, 1e-4, 0.1, 0.95], [1e-5, 1e-4, 0.1, 0.99],
                    [1e-5, 1e-3, 0.01, 0.95], [1e-5, 1e-3, 0.01, 0.99], [1e-5, 1e-3, 0.1, 0.95], [1e-5, 1e-3, 0.1, 0.99],
                    [1e-4, 1e-3, 0.01, 0.95], [1e-4, 1e-3, 0.01, 0.99], [1e-4, 1e-3, 0.1, 0.95], [1e-4, 1e-3, 0.1, 0.99],
                    [1e-4, 1e-4, 0.01, 0.95], [1e-4, 1e-4, 0.01, 0.99], [1e-4, 1e-4, 0.1, 0.95], [1e-4, 1e-4, 0.1, 0.99]]








# Results in attack mode, 100 games
params  0 :  [36. 53. 11.] ###  [1e-5, 1e-4, 0.01, 0.95]
params  1 :  [43. 45. 12.]
params  2 :  [41. 46. 13.] ###  [1e-5, 1e-4, 0.1, 0.95]
params  3 :  [79. 14.  7.]
params  4 :  [53. 33. 14.]
params  5 :  [62. 32.  6.]
params  6 :  [28. 61. 11.] ###  [1e-5, 1e-3, 0.1, 0.95]
params  7 :  [49. 35. 16.]
params  8 :  [32. 53. 15.] ###  [1e-4, 1e-3, 0.01, 0.95]
params  9 :  [38. 38. 24.]
params  10 :  [34. 55. 11.] ###  [1e-4, 1e-3, 0.1, 0.95]
params  11 :  [50. 39. 11.]
params  12 :  [42. 40. 18.]
params  13 :  [53. 34. 13.]
params  14 :  [45. 38. 17.]
params  15 :  [39. 53.  8.] ###   [1e-4, 1e-4, 0.1, 0.99]