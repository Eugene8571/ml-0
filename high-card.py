

deck = ['1A', '2A', '3A', '4A', '5A', '6A', '7A', '8A', '9A', '10A', '11A', '12A', '13A', 
        '1B', '2B', '3B', '4B', '5B', '6B', '7B', '8B', '9B', '10B', '11B', '12B', '13B', 
        '1C', '2C', '3C', '4C', '5C', '6C', '7C', '8C', '9C', '10C', '11C', '12C', '13C', 
        '1D', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', '10D', '11D', '12D', '13D']

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

deck=[[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13]]



def sigmoid(x):
    """
    Функция активации
    """
    return 1 / (1 + 2.718**(-x))


# входящий сигнал - активная карта и число игроков

card = [0,0,0,0,0, 0,0,0,0,0, 0,0,1]
players = [1,1,1,1,1, 0,0,0,0,0]

X = card + players
y = [1]

import numpy

class NeuralNetwork:
    def __init__(self, X, y):
        self.input = X
        self.w1 = numpy.random.rand(len(X), 5)
        self.w2 = numpy.random.rand(5, 1)
        self.y = y

    # def feedforward(self, L1, L2):
    #     self.L1 = 



a = numpy.random.rand(1, len(X))
# print(f"{a=}")

b = numpy.zeros(a.shape)
# print(f"{b=}")

NN = NeuralNetwork(X, y)
# print(f"{NN.w2=}")

d = numpy.dot([[2,1], [1,1]],[[2,2], [1,1]])
print(f"{d=}")

x='5'
print(f"{numpy.array(d).T=}")