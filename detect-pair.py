import numpy as np
import time
from numba import njit

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork(object):
    """
    даёт сигнал, если игрок получил две одинаковых карты
    сет [1,2,3,4,5]*2

    входы:          -
    скрытые слои:   -
    выходы:         -
    

    """


    def __init__(self, X, t):
        self.X = X
        self.t = np.array(t)
        len_H1 = 5
        self.w1 = np.random.rand(len(X), len_H1)
        self.w2 = np.random.rand(len_H1, 2)
        self.b = 0.35


    def feedforward(self):
        self.H1 = np.dot(self.w1.T, self.X)
        self.H1_out = sigmoid(self.H1)
        self.Y = sigmoid(np.dot(self.w2.T, sigmoid(self.H1)))
        Y = self.Y
        t = self.t
        self.C = 1/2*(Y[0]-t[0])**2 + 1/2*(Y[1]-t[1])**2

    def backprop(self):
        r = 0.35

        def get_W0(X, Y, W):
            for y in range(len(Y)):
                for x in range(len(X)):
                    print(f"{range(len(X))=}")
                    print(f"{x=}")
                    print(f"{W=}")
                    w = W[y][x]
                    dw = -(t[y]-Y[y]) * sigmoid_derivative(Y[y]) * X[x]
                    W[y][x] = w - r * dw
            return W

        def get_W1(X, Y, W):
            for y in range(len(Y)):
                for x in range(len(X)):
                    w = W[y][x]
                    dC = -(t[0]-Y[0]) * sigmoid_derivative(Y[0]) * W[y][0] + \
                         -(t[1]-Y[1]) * sigmoid_derivative(Y[1]) * W[y][1]
                    dw = dC * sigmoid_derivative(Y[y]) * X[x]
                    W[y][x] = w - r * dw
            return W



        self.w2 = get_W0(self.H1_out, self.Y, self.w2)
        self.w1 = get_W1(self.X, self.H1, self.w1)






       


X = [0.2,0.01]
t = [1, 0.5]
NN = NeuralNetwork(X,t)
NN.feedforward()
NN.backprop()
i = 0
time1 = time.now()
for x in range(1,100000):
    i += 1 
    if x % 200 == 0:
        print(f"{NN.Y.round(4)=} {NN.C.round(4)=}")
        print(f" ")
    if NN.C < 0.001:
        print(f"{i=} {NN.C=}")
        break
    NN.feedforward()
    NN.backprop()

dt = time.now() - time1
print(f"{dt=}")
