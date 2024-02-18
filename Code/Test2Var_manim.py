from descentAlgorythme import*

def f(point: np.ndarray):
    return point[0]**2 + point[1]**2


print(firstDescent(f,np.array([5,5]),stepDerivative=0.5,stepDescent=0.5,terminationCondition=0.01))