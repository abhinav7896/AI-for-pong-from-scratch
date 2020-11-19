import numpy as np
import matplotlib.pyplot as plt 
import gym
from brain import Brain

class Agent:
    def  __init__(self, nsessions, config):
        if(config['brainModel'] == None):
            self.brain = Brain(config['nInputs'], config['nOutputs'], config['learningRate'], config['dqnConfig'])
        else:
            self.brain = Brain(None, None, None, model=config['brainModel'], dqnConfig = config['dqnConfig'])
        self.nsessions = nsessions
        self.totalReward = 0
        self.rewards = []
        self.epsilon = config['epsilon']
        self.epsilonDecayRate = config['epsilonDecayRate']
        self.batchSize = config['batchSize']
        self.nInputs = config['nInputs']
        # self.env = gym.make('MountainCar-v0').env
        self.env = gym.make("Pong-ram-v0").env
        self.trainingEpochs = config['trainingEpochs']
        self.actionMap = config['dqnConfig']['actionMap']
    
    def train(self):
        session = 0
        UP_ACTION = 2
        DOWN_ACTION = 3
        while(session < self.nsessions):
            session += 1
            self.env.reset()
            currentState = np.zeros((1,self.nInputs))
            nextState = currentState
            gameOver = False
            step = 0
            while not gameOver and step < 5000:   #one episode/session execution
                action = None
                if(np.random.rand() <= self.epsilon):
                    # action = np.random.randint(0, 3)
                    action = np.random.randint(UP_ACTION, DOWN_ACTION+1)
                    print('Exploring...')
                else:
                    qvalues = self.brain.model.predict(currentState)[0]
                    action = self.actionMap[np.argmax(qvalues)]
                    print('Exploiting...')
                nextState[0], reward, gameOver, _ = self.env.step(action)
                self.env.render()
                self.totalReward += reward
                
                #Store experience in DQN Memory
                transition = [currentState, action, reward, nextState]
                self.brain.dqn.storeExp(transition, gameOver)
                #Train
                # experience = self.brain.dqn.retrieveExp(self.batchSize, self.brain)
                # inputs = experience[0]
                # targetqs = experience[1]
                # self.brain.model.train_on_batch(inputs, targetqs)
                currentState = nextState
                print('Step: ', step)
                print('Total Reward', self.totalReward)
                step += 1
                
            # Train from the game session
            print('Training on experience after the episode (a session)....')
            experience = self.brain.dqn.retrieveExp(self.batchSize, self.brain)
            inputs = experience[0]
            targetqs = experience[1]
            # for i in range(self.trainingEpochs):
            self.brain.model.train_on_batch(inputs, targetqs, epochs=self.trainingEpochs)
            
            print('Episode(Session): ' + str(session) +', Epsilon: ' + str(self.epsilon) + ', Total Reward: ' + str(self.totalReward))
            
            #Decay epsilon
            self.epsilon = self.epsilon*self.epsilonDecayRate
            
            #Store the total rewards obtained in this episode/session              
            self.rewards.append(self.totalReward)
            self.totalReward = 0
            plt.plot(self.rewards)
            plt.xlabel('Session (Epsilon: ' + str(self.epsilon) + ')')
            plt.ylabel('Rewards')
            plt.show()
        self.env.close()
           