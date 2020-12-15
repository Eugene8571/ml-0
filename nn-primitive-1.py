import numpy as np


class NeuralNetwork(object):
    """
    один вход, один вес, один выход
    """
    def __init__(self, X, t):
        self.X = X
        self.t = t
        self.w = 0.75
        self.out = None

    def feedforward(self):
        self.out = self.X * self.w

    def backprop(self):
        w = self.w
        r = 0.1
        a = self.out
        X = self.X

        dw = X * (2 * (a - t))

        self.w = w - r * dw


        


X = 1.5
t = 100
NN = NeuralNetwork(X,t)
for x in range(1,10):
    print(f"{NN.out=}")
    NN.feedforward()
    NN.backprop()


