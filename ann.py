import numpy as np
import math

def normalize(input):
  norm = (input - np.min(input, axis=0)) / (np.max(input, axis=0) - np.min(input, axis=0) + 0.001)
  return norm

class ReLU:
    def of(self, input):
        return np.maximum(0.01*input,input)
        # return np.maximum(0,input)
        
    def diff(self,input):
        # return np.maximum(0,input/np.abs(input))
        der = np.zeros(input.shape)
        for i in range(len(input)):
          for j in range(len(input[i])):
            if(input[i][j] < 0):
              der[i][j] = 0.001
            else:
              der[i][j] = input[i][j]
        return der


class Sigmoid:
    def of(self, input):
        return (1/(1+np.exp(-input)))
        
    def diff(self, input):
        s = self.of(input)
        derivativeAt = s*(1-s)
        return derivativeAt

class Softmax:
    def of(self, input):
        # shiftx = input - np.max(input)
        activations = np.exp(input)
        for i in range(len(activations)):
          activations[i] = activations[i] / np.sum(activations[i])
        return activations
        
    def dE(self, input, Y):
        s = self.of(input)
        crossentropydelta = s - Y
        return crossentropydelta

class CrossEntropy:
    def of(self, YHat, Y):
          return -(Y*np.log(YHat) + (1-Y)*np.log(1 - YHat))
        
    def diff(self, YHat, Y):
        derivativeAt = ((1-Y)/(1-YHat)) - Y/YHat
        return derivativeAt

class MSE:
    def of(self, YHat, Y):
          return 0.5*(Y-YHat)**2

class Layer:
    def __init__(self,  units=None, activation=None, **kwargs):
        self.units = units
        self.activation = activation
        self.type = kwargs['type']
        self.layerDelta = None
 
class MLP:
    def __init__(self, architecture=None, optimizer = None, lr=0.01, m=0.9, gamma=0.0, beta1=0.9, beta2=0.999):
        self.optimizer = optimizer
        self.layers = []
        self.W = {}
        self.B = {}
        self.loss = None
        self.metrics = None
        self.NeuronArrays = {}
        self.dW = {}
        self.dB = {}
        self.LEARNING_RATE = lr
        self.MOMENTUM = m
        self.GAMMA = gamma
        self.VdW = {}
        self.SdW = {}
        self.VdB = {}
        self.SdB = {}
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 1
        for layerDescription in architecture:
            layer = Layer(units=layerDescription['units'], activation=layerDescription['activation'], type=layerDescription['type'])
            self.layers.append(layer)
        self.__populateSynapses()
            
    def __populateSynapses(self):
        for i in range(len(self.layers)-1):
            if(self.layers[i].type == 'dense' or self.layers[i].type == 'input'):
                synapses = np.random.rand(self.layers[i+1].units, self.layers[i].units)*np.sqrt(1/self.layers[i].units) # Xavier initialization
                self.W[(i, 'current')] = synapses
                self.dW[i] = np.zeros(self.W[(i, 'current')].shape)
                self.B[i+1] = np.random.rand(1, self.layers[i+1].units) #*np.sqrt(1/self.layers[i].units)           # Xavier initialization
                self.dB[i+1] = np.zeros(self.B[i+1].shape)
                if(self.optimizer == 'adam'):
                   self.VdW[i] = np.zeros(self.W[(i, 'current')].shape)
                   self.SdW[i] = np.zeros(self.W[(i, 'current')].shape)
                   self.VdB[i+1] = np.zeros(self.B[i+1].shape)
                   self.SdB[i+1] = np.zeros(self.B[i+1].shape)

                
    # def compileNN(metrics=None, optimizer=None, loss=MSE(), **kwargs):
    #     self.metrics = metrics
    #     self.optimizer = optimizer
    #     self.loss = loss
    #     if(self.layers[len(self.layers)-1].activation == 'softmax'):
    #       self.loss = CrossEntropy()
        
    def train_on_batch(self, X_train, Y_train=None, X_test=None, Y_test=None, batchSize = None, stochastic=False, epochs=10):
        errorsTrain = []
        errorsTest = []
        normalize = Sigmoid()
        loss = MSE()      
        for i in range(epochs):
            print('Training')
            self.__feedForward(X_train)
            YHat = self.NeuronArrays[len(self.layers)-1]['actionPotentials']
            self.__initiateBackpropagation(YHat, Y_train)
            for j in range(len(self.layers)-2, 0, -1):
                self.__backpropagate(j, self.layers[j+1].layerDelta, self.W[(j, 'previous')], self.layers[j].activation)
            trainerror = np.sum(np.abs(YHat - Y_train))/X_train.shape[0]
            print("Epoch: {}, Train Error: {}".format(i, trainerror))
            errorsTrain.append(trainerror.astype(float))
            # testerror = self.predict(X_test, Y_test)[1]
            # errorsTest.append(testerror.astype(float))
            self.t = self.t + 1
        # print(YHat)

        return (errorsTrain, errorsTest)
    
    def predict(self, X=None, Y=None):
      # print(self.W[((0), 'current')].shape)
      self.__feedForward(X)
      YHat = self.NeuronArrays[len(self.layers)-1]['actionPotentials']
      # error = np.sum(np.abs(YHat - Y))/X.shape[0]
      print('YHat is')
      return (YHat, 0)
    
    def __feedForward(self, X):
        # if(X.shape[1] != self.layers[0].units):
        #     print("Input shape mismatch!", X.shape[0], self.layers[0].units)
        # else:
          Z = {}
          Z['inputs'] = X                                      #Input layer values
          Z['actionPotentials'] = Z['inputs'].copy()           #Input layer has no activation. Hence activation ouptut = input
          Z['activation'] = None
          Z['actionPotentials'] = normalize(Z['actionPotentials'])
          self.NeuronArrays[0] = Z.copy()            
          activation = ReLU()
          for i in range(1, len(self.layers)):
              if(self.layers[i].activation == 'sigmoid'):
                  # print('Sigmoid activation')
                  activation = Sigmoid()
                  Z['activation'] = activation
              elif(self.layers[i].activation == 'softmax'):
                  # print('Softmax activation')
                  activation = Softmax()
                  Z['activation'] = activation
              else:
                  print("Default activation: ReLU")
              Z['biases'] = self.B[i].copy()
              # print("Biases shape", self.B[i].shape)
              Z['inputs'] = np.dot(Z['actionPotentials'], self.W[((i-1), 'current')].T ) + Z['biases']          
              Z['actionPotentials'] = activation.of(Z['inputs'])
              # Z['actionPotentials'] = normalize(Z['actionPotentials'])
              self.NeuronArrays[i] = Z.copy()
                
    
    def __initiateBackpropagation(self, YHat, Y_train):
        print('Determining output delta..')
        lastLayer = len(self.layers)-1
        errorDelta = None
        if(self.layers[lastLayer].activation == 'softmax'):
          print('Softmax, Cross Entropy')
          activation = Softmax()
          errorDelta = activation.dE(self.NeuronArrays[lastLayer]['inputs'], Y_train)
        elif(self.layers[lastLayer].activation == 'relu'):
          print('ReLU')
          activation = ReLU()
          errorDelta = -1*(Y_train-YHat)*activation.diff(self.NeuronArrays[lastLayer]['inputs'])
        else:
          errorDelta = -1*(Y_train-YHat)*YHat*(1-YHat) # error derivative times the last layer activation's derivative
        # self.layers[lastLayer].layerDelta = np.vstack([np.sum(errorDelta, axis=0)/Y_train.shape[0]]*Y_train.shape[0])
        self.layers[lastLayer].layerDelta = errorDelta
        self.dW[lastLayer-1] = self.MOMENTUM * self.dW[lastLayer-1] + np.dot(self.layers[lastLayer].layerDelta.T, self.NeuronArrays[lastLayer-1]['actionPotentials'])
        self.dB[lastLayer] = self.MOMENTUM * self.dB[lastLayer] + self.layers[lastLayer].layerDelta[0]
        if(self.optimizer == 'adam'):
          print('Starting Adam optimization.............')
          self.adam(lastLayer)
          return
        self.W[(lastLayer-1, 'previous')] = self.W[(lastLayer-1, 'current')].copy()
        self.W[(lastLayer-1), 'current'] = (1-2*self.LEARNING_RATE*self.GAMMA)*self.W[(lastLayer-1), 'current'] - self.LEARNING_RATE*self.dW[lastLayer-1]   # L2 regularization (weight decay / ridge regression)
      #  bias deltas
        self.B[lastLayer] = self.B[lastLayer] - self.LEARNING_RATE*self.dB[lastLayer]

    def __backpropagate(self, currentLayerIndex, backpropagatedDelta, weights, layerActivation):
        activation = ReLU()
        print('Backpropagating...')
        if(layerActivation == 'sigmoid'):
            # print('Sigmoid activation derivative')
            activation = Sigmoid()
        if(layerActivation == 'softmax'):
            # print('Softmax activation derivative')
            activation = softmax()
        derivative = activation.diff(self.NeuronArrays[currentLayerIndex]['inputs'])
        # print(derivative)
        self.layers[currentLayerIndex].layerDelta = np.dot(backpropagatedDelta, weights)*derivative
        self.dW[currentLayerIndex - 1] = self.MOMENTUM * self.dW[currentLayerIndex - 1] + np.dot(self.layers[currentLayerIndex].layerDelta.T, self.NeuronArrays[currentLayerIndex-1]['actionPotentials'])
        self.dB[currentLayerIndex] = self.MOMENTUM * self.dB[currentLayerIndex] + self.layers[currentLayerIndex].layerDelta[0]
        if(self.optimizer == 'adam'):
          print('Starting Adam optimization.............')
          self.adam(currentLayerIndex)
          return
        self.W[(currentLayerIndex-1, 'previous')] = self.W[(currentLayerIndex-1, 'current')].copy()
        self.W[(currentLayerIndex-1, 'current')] = (1-2*self.LEARNING_RATE*self.GAMMA)*self.W[(currentLayerIndex-1, 'current')] - self.LEARNING_RATE*self.dW[currentLayerIndex-1]    # L2 regularization (weight decay / ridge regression)
        self.B[currentLayerIndex] = self.B[currentLayerIndex] - self.LEARNING_RATE*self.dB[currentLayerIndex]

    def adam(self, currentLayerIndex):
      print('Using Adam optimizer....................')
      self.VdW[currentLayerIndex - 1] = self.beta1 * self.VdW[currentLayerIndex - 1] + (1-self.beta1)*self.dW[currentLayerIndex - 1]
      self.VdB[currentLayerIndex] = self.beta1 * self.VdB[currentLayerIndex] + (1-self.beta1)*self.dB[currentLayerIndex]
      self.SdW[currentLayerIndex - 1] = self.beta2 * self.SdW[currentLayerIndex - 1] + (1-self.beta2)*self.dW[currentLayerIndex - 1]**2
      self.SdB[currentLayerIndex] = self.beta2 * self.SdB[currentLayerIndex] + (1-self.beta2)*self.dB[currentLayerIndex]**2
      VdWc = self.VdW[currentLayerIndex - 1]/(1-self.beta1**self.t)
      VdBc = self.VdB[currentLayerIndex]/(1-self.beta1**self.t)
      SdWc = self.SdW[currentLayerIndex - 1]/(1-self.beta2**self.t)
      SdBc = self.SdB[currentLayerIndex]/(1-self.beta2**self.t)
      eps = 0.000001
      stepW = VdWc/np.sqrt(SdWc + eps)
      stepB = VdBc/np.sqrt(SdBc + eps) 
      self.W[(currentLayerIndex-1, 'previous')] = self.W[(currentLayerIndex-1, 'current')].copy()
      self.W[(currentLayerIndex-1, 'current')] = (1-2*self.LEARNING_RATE*self.GAMMA)*self.W[(currentLayerIndex-1, 'current')] - self.LEARNING_RATE*stepW    # L2 regularization (weight decay / ridge regression)
      self.B[currentLayerIndex] = self.B[currentLayerIndex] - self.LEARNING_RATE*stepB

