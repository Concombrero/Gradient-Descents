from firstDescent import*
from momentumDescent import*

def f(X):
    return X[0]**2+X[1]**2

i=np.array([16.0,16.0])

print(momentumGradientDescent(f,i,0.1,0.1,1))