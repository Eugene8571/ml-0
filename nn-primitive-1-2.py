import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

class NeuralNetwork(object):
    """
    один вход,
    один скрытый слой,
    два веса, 
    один выход
    """
    def __init__(self, X, t):
        self.X = X
        self.t = t
        self.w1 = 0.75
        self.w2 = 0.33
        self.out = None

    def feedforward(self):
        self.H1 = self.X * self.w1
        self.H1_out = sigmoid(self.H1)
        self.out = self.H1_out * self.w2

    def backprop(self):
        w1 = self.w1
        w2 = self.w2
        r = 0.001
        a = self.out
        X = self.X
        H1 = self.H1


        dw2 = H1 * (2 * (a - t))
        self.w2 = w2 - r * dw2
        dw1 = X * w2 * (2 * (a - t))
        self.w1 = w1 - r * dw1
        


X = 1.5
t = 1
NN = NeuralNetwork(X,t)
for x in range(1,2000):
    print(f"{NN.out=}")
    print(f"{NN.w1=}")
    print(f"{NN.w2=}")
    NN.feedforward()
    NN.backprop()