import numpy as np

from TD3 import agent
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client

class RemoteTD3(agent.TD3, RemoteControllerInterface):

    def __init__(self, pretrained='stronger'):
        agent.TD3.__init__(self, pretrained=pretrained)
        RemoteControllerInterface.__init__(self, identifier='StillTrying-TD3')

    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:

        return self.act(obs)
        

if __name__ == '__main__':
    controller = RemoteTD3()

    # Play n (None for an infinite amount) games and quit
    client = Client(username='Fynn_Bachmann_StillTrying', # Testuser
                    password="W@7c'M{Z",
                    controller=controller, 
                    output_path='/tmp/ALRL2020/client/user0', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=None)

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queuing
    # client = Client(username='user0', 
    #                 password='1234',
    #                 controller=controller, 
    #                 output_path='/tmp/ALRL2020/client/user0',
    #                )
