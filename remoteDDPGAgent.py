import numpy as np
from DDPG import ddpg_agent
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client

class RemoteDDPG(ddpg_agent.DDPGAgent, RemoteControllerInterface):

    def __init__(self, pretrained='DDPG/weights/ddpg-newest-td3-4000'):
        ddpg_agent.DDPGAgent.__init__(self, pretrained=pretrained)
        RemoteControllerInterface.__init__(self, identifier='StillTrying-DDPG')

    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:

        return self.act(obs)
        

if __name__ == '__main__':
    controller = RemoteDDPG()

    # Play n (None for an infinite amount) games and quit
    client = Client(username='Timo_Maier_DDPG_StillTrying', # Testuser
                    password='UCk})2D`',
                    controller=controller, 
                    output_path='/tmp/ALRL2020/client/Timo_Maier_DDPG_StillTrying', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=None)

    # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queuing
    # client = Client(username='user0', 
    #                 password='1234',
    #                 controller=controller, 
    #                 output_path='/tmp/ALRL2020/client/user0',
    #                )
