from gradientFunction import*
from descentAlgorithme import*

def f(X):
    return X[1]**3-np.exp(-1*X[0]**2)
    
path=firstDescent(f, np.array([2., 0.]), 0.1, 0.1, 0.01)
liste=[]
gradients=[]

for point in path:
    liste.append(point[0])
    gradients.append(point[1])

points = [tuple(arr) for arr in liste]
liste_finale=list()
for point in points:
    x=round(point[0], 3)
    y=round(point[1], 3)
    liste_finale.append((x,y,round(f(np.array([x,y])),3)))

grads = [tuple(arr) for arr in gradients]
grad=[]   
for gradient in grads:
    x=round(gradient[0], 3)
    y=round(gradient[1], 3)
    grad.append((x,y))
    
print(liste_finale)
print(grad)