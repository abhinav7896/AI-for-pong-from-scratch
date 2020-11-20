import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
tf.config.list_physical_devices('GPU')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import os
#Set order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    
#Select gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

#The Brain with a deep-q network

class Brain:
    def __init__(self, nInputs, nOutputs, learningRate, model=None, dqnConfig=None):
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.learningRate = learningRate
        if(model == None):
            self.model = Sequential()
            self.model.add(Dense(units=64, activation='relu', input_shape=(self.nInputs,)))
            self.model.add(Dense(units=16, activation='relu'))
            self.model.add(Dense(units=8, activation='relu'))
            self.model.add(Dense(units=nOutputs))
            self.model.compile(optimizer=Adam(lr=self.learningRate), loss='mean_squared_error')
        else:
            self.model = model
            self.nInputs = model.layers[0].units
            self.nOutputs = model.layers[-1].units
            self.learningRate = learningRate
        if(dqnConfig != None):
            self.dqn = self.DeepQNetwork(dqnConfig)

    class DeepQNetwork:
        def __init__(self, dqnConfig):
            self.maxMemory = dqnConfig['maxMemory']
            self.gamma = dqnConfig['discount']
            self.dqnMemory = list()
            self.invActionMap = dqnConfig['invActionMap']
        
        def storeExp(self, transition, gameOver):
            self.dqnMemory.append([transition, gameOver])
            if(len(self.dqnMemory) > self.maxMemory):
                self.dqnMemory.pop(0)
        
        def retrieveExp(self, batchSize, brain):
            nInputs = brain.nInputs
            nOutputs = brain.nOutputs
            inputs = np.zeros((min(batchSize, len(self.dqnMemory)), nInputs))
            targetqs = np.zeros((min(batchSize, len(self.dqnMemory)), nOutputs))
            i = 0
            for inx in np.random.randint(0, len(self.dqnMemory), size=min(batchSize, len(self.dqnMemory))):
                currentState, action, reward, nextState = self.dqnMemory[inx][0]
                gameOver = self.dqnMemory[inx][1]
                inputs[i] = currentState
                targetqs[i] = brain.model.predict(currentState)[0]   #get initial arbitrary output to be replaced by the values from bellman equation
                #Update target q-values with calculated q-values from bellman equation for q-learning
                if(gameOver):
                    targetqs[i][self.invActionMap[action]] = reward
                else:
                    targetqs[i][self.invActionMap[action]] = reward + self.gamma*np.max(brain.model.predict(nextState)[0])
                i += 1
            return (inputs,targetqs)