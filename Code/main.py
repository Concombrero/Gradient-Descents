from gradientFunction import*
from descentAlgorythme import*

def f1(X):
    return X[0]**2

def f2(X):
    return X[0]**2+X[1]**2+X[0]*X[1]

f1= lambda X: X[0]**2
f2=lambda X: X[0]**2+X[1]**2+X[0]*X[1]

print(len(firstDescent(f1, np.array([14774.0]), 0.1, 0.1, 0.001)))
print(len(heavyBallDescent(f1, np.array([14774.0]), 0.1, 0.1, 0.001, 0.5)))
print(len(nesterovDescent(f1, np.array([14774.0]), 0.1, 0.1, 0.001)))