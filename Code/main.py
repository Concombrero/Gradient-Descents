from gradientFunction import*
from descentAlgorythme import*

def f1(X):
    return X[0]**2

def f2(X):
    return X[0]+X[1]**2

print(firstDescent(f1, np.array([16.0]), 0.1, 0.1, 1))