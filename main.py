from laserhockey.hockey_env import HockeyEnv, BasicOpponent
from laserhockey.gameplay import gameplay
from DQN import agent
from DDPG.ddpg_agent import DDPGAgent
from TD3.agent import TD3

env = HockeyEnv()

basic = BasicOpponent(weak=False)
weak = BasicOpponent(weak=True)

ddpg = DDPGAgent(pretrained="DDPG/weights/checkpoint4")

q_agent = agent.DQNAgent(env.observation_space, env.discrete_action_space,
                        convert_func =  env.discrete_to_continous_action,
                        pretrained   = 'DQN/weights/alg2')

# td3 = TD3(pretrained='superagent')
td3 = TD3(pretrained='lasttry')

stats = gameplay(env, td3, player2=weak, N=100, show=False, analyze=False)
print(stats)

env.close()
