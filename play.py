import numpy as np

X = [0,1]
w1 = [[-0.5, 0.1],
    [0.2, 0.4]]

L1 = np.dot(X,w1)
print(f"{L1=}")

w2 = [0.3, 0.7]
out = np.dot(L1, w2)
print(f"{out=}")

d_out = 1 - 0.34
print(f"{d_out=}")

d_L1 = np.dot(d_out, w2)
print(f"{d_L1=}")

print(f"{0.66*0.3=}")
print(f"{0.66*0.7=}")


a=np.array([0.63870761, 0.68497694])[np.newaxis].T
b=np.array([0.14006937])

# r=np.dot(a.reshape(2,1),b)
r = np.dot(a,b)


# print(f"{r=}")

