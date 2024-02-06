from firstDescent import*

def f(X):
    return X[0]**2+X[1]**2

i=np.array([16.0,16.0])

print(firstDescent(f,i,0.1,0.1,0.5))