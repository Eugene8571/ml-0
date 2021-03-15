'''
игрок со старшей картой побеждает.

в колоде 10 карт, игроки получают по одной
[1,2,3,4,5,6,7,8,9,10]

'''

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

r = 0.1


class NeuralNetwork(object):
    """docstring for NeuralNetwork"""
    def __init__(self, X, t):
        self.input = X
        self.t = t
        self.w1=np.array([ [0.34740112, 0.71587649, 0.54477065],
                           [0.12048776, 0.80215174, 0.7848461 ],
                           [0.6657934 , 0.99960724, 0.53942614],
                           [0.72586272, 0.6453454 , 0.76229148],
                           [0.84048451, 0.41317105, 0.5744454 ],
                           [0.32745197, 0.42230799, 0.85048843],
                           [0.32768473, 0.83817198, 0.81213001],
                           [0.29800999, 0.11962701, 0.60332183],
                           [0.07127859, 0.04723507, 0.58615689],
                           [0.08824525, 0.77525399, 0.53755418]])

        self.w2=np.array([ [0.95994211, 0.59427341],
                           [0.27699257, 0.15645139],
                           [0.05190993, 0.23507123]])

        self.w3=np.array([ [0.59027993, 1.25091387],
                           [0.65712936, 1.18164082]])

        self.out = None
        self.E = None


    def feedforward(self):
        self.out_L1 = sigmoid(np.dot(self.input, self.w1))
        self.out_L2 = sigmoid(np.dot(self.out_L1, self.w2))
        self.out = np.dot(self.out_L2, self.w3)
        return self.out

    def backprop(self):
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        X = self.input
        out_L1 = self.out_L1
        out_L2 = self.out_L2
        out = self.out
        t = self.t

        E_total = 1/len(out) * np.sum((out - t)**2)
        E_total = 1/2 * (out[0]-t[0])**2 + 1/2 * (out[1]-t[1])**2
        self.E = E_total
        r = 0.3 

        w = w3[0][0]
        t = self.t[0]
        out_y = out[0]
        dw = -(t-out_y) * out_y*(1-out_y) * out_L1[0]
        w3[0][0] = w - r * dw

        w = w3[0][1]
        t = self.t[0]
        out_y = out[0]
        dw = -(t-out_y) * out_y*(1-out_y) * out_L1[1]
        w3[0][1] = w - r * dw

        w = w3[1][0]
        t = self.t[1]
        out_y = out[1]
        dw = -(t-out_y) * out_y*(1-out_y) * out_L1[1]
        w3[1][0] = w - r * dw

        w = w3[1][1]
        t = self.t[1]
        out_y = out[1]
        dw = -(t-out_y) * out_y*(1-out_y) * out_L1[0]
        w3[1][1] = w - r * dw

        self.w3 = w3

        return

        


X = [0,0,0,0,0,0,0,0,0,1]
t = [1,9]
NN = NeuralNetwork(X,t)
print(f"{NN.feedforward()=}")
for i in range(1000):
    if i%200 == 0:
        print(f"{NN.out=}")
        print(f"{NN.E=}")
    NN.feedforward()
    NN.backprop()

print(f"{NN.w3=}")

