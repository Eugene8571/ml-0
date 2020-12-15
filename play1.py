import numpy as np


def sigmoid(t):
    return 1/(1+np.exp(-t))

def sigmoid_derivative(t):
    return t * (1 - t)

class NeuralNetwork:
    """docstring for NeuralNetwork"""
    def __init__(self, X, y):
        self.i = X
        self.y = y
        # self.w1 = np.random.rand(len(X), 3).round(2)
        self.w1 = np.array([[0.77, 0.99, 0.92],
                            [0.56, 0.13, 0.96 ]])
        # self.w2 = np.random.rand(3, 2)
        self.w2 = np.array([[0.03, 0.25],
                            [0.96, 0.96],
                            [0.88, 0.46]])
        # self.w3 = np.random.rand(2, 1).round(2)
        self.w3 = np.array([[0.19],
                            [0.34]])
    

    def feedforward(self):
        self.L1 = sigmoid(np.dot(X, self.w1)).round(2)
        self.L2 = sigmoid(np.dot(self.L1, self.w2)).round(2)
        self.out = sigmoid(np.dot(self.L2, self.w3)).round(2)
        return self.out

    def backprop(self):
        self.C = ((self.y - self.out)**2)
        d = self.C * sigmoid_derivative(self.out) # локальный градиент выходного нейрона
        h = 2 # шаг сходимости
        d_w3 = np.dot(self.L2, h*d*sigmoid_derivative(self.L2))
        self.w3 += d_w3
        d_w2 = np.dot(self.L1, h*d*sigmoid_derivative(self.L2), self.L2, )
        print(f"{d_w2=}")

        return


X = [0,1]
y = [1]
NN = NeuralNetwork(X,y)
NN.feedforward()
NN.backprop()

