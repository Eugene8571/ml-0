import numpy as np


def sigmoid(t):
    return 1/(1+np.exp(-t))

class NeuralNetwork:
    """docstring for NeuralNetwork"""
    def __init__(self, X, y):
        self.i = X
        self.y = y
        # self.w1 = np.random.rand(len(X), 3)
        self.w1 = np.array([[0.84, 0.65, 0.04],
                            [0.72, 0.82, 0.25],
                            [0.77, 0.99, 0.92],
                            [0.56, 0.13, 0.96 ]])
        # self.w2 = np.random.rand(3, 2)
        self.w2 = np.array([[0.03, 0.25],
                            [0.96, 0.96],
                            [0.88, 0.46]])
        # self.w3 = np.random.rand(2, 2).round(2)
        self.w3 = np.array([[0.8 , 0.29],
                            [0.95, 0.38]])
    

    def feedforward(self):
        self.L1 = sigmoid(np.dot(X, self.w1)).round(2)
        print(f"{self.L1=}")
        self.L2 = sigmoid(np.dot(self.L1, self.w2)).round(2)
        print(f"{self.L2=}")
        self.out = sigmoid(np.dot(self.L2, self.w3)).round(2)
        return self.out

    def backprop(self):
        self.C = ((self.y - self.out)**2).round(2)
        print(f"{self.y=}")
        print(f"{self.C=}")
        return 'backprop'


X = [0,1,0,0]
y = [1,0]
NN = NeuralNetwork(X,y)
print(f"{NN.feedforward()=}")
NN.backprop()

