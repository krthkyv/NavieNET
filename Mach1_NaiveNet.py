import random
import math

class value:
    def __init__(self, data, label = "V", backprop = lambda : None, grad = 0.0):
        self.data, self.label, self.backprop, self.grad = data, label, backprop, grad
    def __add__(self, other):
        out =  value(self.data + other.data)
        def backprop():
            self.grad += out.grad
            other.grad += out.grad
            self.backprop()
            other.backprop()
        out.backprop = backprop
        return out
    def __mul__(self, other):
        out =  value(self.data * other.data)
        def backprop():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            self.backprop()
            other.backprop()
        out.backprop = backprop
        return out
    def __radd__(self, other):
        return self.__add__(value(other))
    def __repr__(self):
        return f"{self.label} : {self.data}"
    def __pow__(self, power : int):
        out = value(self.data ** power)
        def backprop():
            self.grad+= out.grad * (power * (self.data ** (power - 1)))
            self.backprop()
        out.backprop = backprop
        return out
    def __sub__(self, other):
        return self + (value(-1) * other)
    def __truediv__(self, other):
        return self * (other ** -1)
    def ReLU(self):
        out = value(0 if self.data < 0 else self.data)
        def backprop():
            self.grad += out.grad * (0 if self.data < 0 else 1)
            self.backprop()
        out.backprop = backprop
        return out
    
    def tanh(self):
        out = value(((math.exp(2 * self.data) - 1 + 1e-15) / (math.exp(2 * self.data) + 1) + 1e-15))
        def backprop():
            self.grad += out.grad * (1 - (out.data ** 2))
            self.backprop()
        out.backprop = backprop
        return out

def Numerical2Value(numericalArr : list) -> list[value]:
    return [value(numerical) for numerical in numericalArr]

def NumArrs2ValArrs(ArrOfNum : list[list[float]]) -> list[list[value]]:
    return [Numerical2Value(numArr) for numArr in ArrOfNum]

def mse(predictedVec : list[float], actualVec : list[float]) -> value:
    if isinstance(predictedVec[0], float) or isinstance(predictedVec[0], int):
        predictedVec = Numerical2Value(predictedVec)
    
    if isinstance(actualVec[0], float) or isinstance(actualVec[0], int):
        actualVec = Numerical2Value(actualVec)
    
    if not isinstance(actualVec[0], value):
        raise TypeError

    if not isinstance(predictedVec[0], value):
        raise TypeError
    
    simpleError = sum([(prediction - actual) ** 2 for prediction , actual in zip(predictedVec, actualVec)]) / value(len(actualVec))
    return simpleError

class neuron:
    def __init__(self, inputSize : int):
        self.b = value(random.uniform(-1, 1))
        self.w = [value(random.uniform(-1, 1)) for _ in range(inputSize)]
    def __call__(self, inputs):
        weightedSum = sum([xi * wi for xi, wi in zip(inputs, self.w)])
        addedBias = self.b + weightedSum
        activated = addedBias.tanh()
        return activated
    def getParams(self):
        return [self.b] + self.w

class layer:
    def __init__(self, inputSize, outputSize):
        self.neurons = [neuron(inputSize) for _ in range(outputSize)]
    def __call__(self, inputs):
        return [n(inputs) for n in self.neurons]
    def getParams(self):
        return [params for neuron in self.neurons for params in neuron.getParams()]

class network:
    def __init__(self, layerDims : list):
        self.layers = []
        for i in range(len(layerDims) - 1):
            self.layers.append(layer(layerDims[i], layerDims[i + 1]))
    def __call__(self, inputVec : list[float]):
        if isinstance(inputVec[0], float) or isinstance(inputVec[0], int):
            inputVec = Numerical2Value(inputVec)

        if not isinstance(inputVec[0], value):
            raise TypeError
        
        outputVec = inputVec
        for layer in self.layers:
            outputVec = layer(outputVec)
        return outputVec
    def getParams(self) -> list[value]:
        return [params for layer in self.layers for params in layer.getParams()]

def trainer(network, xTrain, yTrain, lr = 0.01, epochs = 1000):
    for epochs in range(epochs):
        for x, y in zip(xTrain, yTrain):
            pred, act = network(x), y
            loss = mse(pred, act)
            def flushGradients():
                for p in network.getParams():
                    p.grad = 0.0
            
            def backpropLoss():
                loss.grad = 1.0
                loss.backprop()
            
            def clipGrad():
                for p in network.getParams():
                    if p.grad > 7:
                        p.grad = 7
                    if p.grad < -7:
                        p.grad = -7

            def adjustParams():
                for p in network.getParams():
                    p.data -= p.grad * lr
            

            flushGradients()
            backpropLoss()
            clipGrad()
            adjustParams()

def getNetwork(layerDims : list[int]) -> network:
    try:
        n = network(layerDims)
        return n
    except:
        return None
    
def trainNet(network : network, xTrain : list[list[float]], yTrain : list[list[float]], lr : float, epochs : int) -> bool:
    try:
        xTrain, yTrain = NumArrs2ValArrs(xTrain), NumArrs2ValArrs(yTrain)
        trainer(network, xTrain, yTrain, lr, epochs)
        return True
    except:
        return False

def dataSetErrorEval(x : list[list[float]], y : list[list[float]], network) -> float:
    p = [network(xi) for xi in x]
    mae = 0
    for act, pred in zip(y, p):
        mae+=abs(act[0] - pred[0].data)
    return mae / len(x)




