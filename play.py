import numpy as np

def f(x):
    return 1/(1+np.exp(-x))


w = np.array([[0, 0],
              [0, 0],
              [0, 0]])

X = [1,2,3]

Y = np.dot(X,w)

# print(f"{Y=}")


a = np.array([[2,2],[5,5]])
b = np.array([3,3])
c = np.array([[5]])



Y=np.array([0.9892002 , 0.49998858])
t=np.array([1. , 0.5])

Y=np.array([0.98635056, 0.50218719])
t=np.array([1. , 0.5])
w2=np.array([[0.81238427, 1.05046937],
       [0.34072156, 0.6107213 ]])
w1=np.array([[0.79662004, 0.72483148],
       [0.37789945, 0.50889507]])
# NN.C=9.554555711669237e-05

E = (Y[0]-t[0])**2 + (Y[1]-t[1])**2

print(f"{'qwerty'[::-1]=}")