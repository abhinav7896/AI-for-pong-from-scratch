#See devices
import os        
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#Set order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
#Select gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from agent import Agent
from ann import MLP


arch = [{'units': 128, 'activation': None, 'type': 'input'},
     {'units': 64, 'activation': 'relu', 'type': 'dense'},
     {'units': 2, 'activation': 'relu', 'type': 'dense'}]

brainModel = MLP(arch,'adam', lr=0.01, m=0.0, gamma=0.009, beta1=0.9, beta2=0.99)

config = {
    'batchSize': 1024,
    'brainModel': brainModel,
    'nInputs': 128, 
    'nOutputs': 2, 
    'learningRate':0.01, 
    'dqnConfig':
          {
              'maxMemory': 50000,
              'discount': 0.9,
              'actionMap': {
                  2: 0,
                  3: 1,
                  0: 2,
                  1: 3
                  }
          },
    'trainingEpochs': 550,
    'epsilon': 1.0,
    'epsilonDecayRate': 0.995
}
nsessions = 10000   #Number of episodes or sessions to be played and trained on in total
agent = Agent(nsessions, config)
agent.train()