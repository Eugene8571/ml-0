

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



def sigmoid(t):
    return 1/(1+np.exp(-t))

def sigmoid_derivative(p):
    return p * (1 - p)


'''
входящий сигнал - активная карта из 13-ти
два игрока
побеждает игрок со старшей картой.

'''

card = [0,0,0,0,0, 0,0,0,0,0, 0,0,1]

X = card

import numpy as np

class NeuralNetwork:
    def __init__(self, X, y):
        self.input = X
        self.w1 = np.random.rand(len(X), 2)
        self.w2 = np.random.rand(2, 1)
        self.y = y
        self.output = np.zeros(y.shape)



    def feedforward(self):
        """
        Функция для прямого прохождения между двумя слоями нейронов
        L1 входящий слой
        L2 выходящий
        W  связи между ними

        1 По массиву W определить количество входных и выходных нейронов

        2.1 Цикл по выходным нейронам, в начале каждой итерации обнуляеть выходной нейрон
        2.2 Вложенный цикл по входным нейронам. 
        Умножать их значения на веса соответствующих связей и складывать
        Результат пропустить через функцию активации, получая новое значение выходного нейрона
        """
        self.layer1 = sigmoid(np.dot(self.input, self.w1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.w2))
        return self.layer2


    def backprop(self):
        """
        """
        print(f"{self.layer1.T=}")
        print(f"{2*(self.y -self.output)*sigmoid_derivative(self.output)=}")


        d_w2 = np.dot(self.layer1.T, 
            2*(self.y -self.output)*sigmoid_derivative(self.output))
        d_w1 = np.dot(self.input.T, 
            np.dot(2*(self.y -self.output)*sigmoid_derivative(self.output), self.w2.T)*sigmoid_derivative(self.layer1))
    
        self.w1 += d_w1
        self.w2 += d_w2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()


X = np.array([0,0,0,0,0, 0,0,0,0,0, 0,0,1], dtype=float)
y = np.array([1], dtype=float)

NN = NeuralNetwork(X, y)
for i in range(1500): # trains the NN 1,000 times
    pass
    if i % 500 ==0: 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) # mean sum squared loss
        print ("\n")
  
    NN.train(X, y)

