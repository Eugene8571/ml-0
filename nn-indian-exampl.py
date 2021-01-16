import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork(object):
    """
    входы:          2
    скрытые слои:   1x2
    выходы:         2
    

    """


    def __init__(self, X, t):
        self.X = X
        self.t = t
        self.w1 = np.array([[0.15, 0.20],
                            [0.25, 0.30]])
        self.w2 = np.array([[0.40, 0.45],
                            [0.50, 0.55]])
        self.b1 = 0.35
        self.b2 = 0.60
        self.Y = None
        self.H = None
        self.E = None

    def feedforward(self):
        '''
        H1 = X1*w1 + X2*w2 + b1
        out_H1 = sigmoid(H1)
        E_total = 1/2 * (t1 - out_y1)**2 + 1/2 * (t2 - out_y2)**2 

        '''
        self.H1 = np.dot(self.X, self.w1.T) + self.b1
        self.H1_out = sigmoid(self.H1)
        self.Y = np.dot(self.H1_out, self.w2.T) + self.b2
        self.Y_out = sigmoid(self.Y)
        print(f"{self.Y=}")
        print(f"{self.Y_out=}")
        Y_out = self.Y_out
        t = self.t
        self.E = 1/2*(Y_out[0]-t[0])**2 + 1/2*(Y_out[1]-t[1])**2
        print(f"{self.E=}")

    def backprop(self):
        '''
        dE_total / dout_y1 = 2 * 1/2 * (t1 - out_y1)**(2-1) * (-1) 
                           = - (t1 - out_y1)
        '''
        t1 = self.t[0]
        out_y1 = self.Y_out[0]
        dE_dout_y1 = - (t1 - out_y1)
        print(f"{dE_dout_y1=}")

        # dout_y1_dy1 = 



       


X = [0.05, 0.10]
t = [0.01, 0.99]
NN = NeuralNetwork(X,t)
NN.feedforward()
NN.backprop()
# for x in range(1,10):
#     if x % 50 == 0:
#         print(f"{NN.Y=}")
#         print(f"{NN.w2=}")
#         print(f"{NN.w1=}")
#         # print(f"{NN.C=}")
#     NN.feedforward()
#     NN.backprop()