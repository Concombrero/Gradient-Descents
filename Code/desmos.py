from gradientFunction import*
from descentAlgorithme import*

def f1(X):
    return -1/(X[0]**2 + X[1]**2 + 3) + 1/((X[0]-4)**2 + (X[1]-1)**2 + 1) -1/((X[0]+4)**2 + (X[1]+1)**2 + 2/3)


paths=[heavyBallDescent(f1, np.array([-6.0,-4.0]), 0.1, 0.1, 0.01,0.7,0.7), heavyBallDescent(f1, np.array([3.0,1.0]), 0.1, 0.1, 0.01,0.7,0.7)]
i=0
gradients_finale=[[] for k in range(len(paths))] 
points_finale=[[] for k in range(len(paths))]

for path in paths:

    liste=[]
    gradients=[]

    for point in path:
        liste.append(point[0])
        gradients.append(point[1])

    points = [tuple(arr) for arr in liste]
    
    for point in points:
        x=round(point[0], 3)
        y=round(point[1], 3)
        points_finale[i].append((x,y,round(f1(np.array([x,y])),2)))

    grads = [tuple(arr) for arr in gradients]
    for gradient in grads:
        x=round(gradient[0], 2)
        y=round(gradient[1], 2)
        gradients_finale[i].append((x,y)) 
    
    i+=1

    
print(gradients_finale)
with open('desmos.txt','w') as f:
 for k in range(len(points_finale)):
    f.write(f"P_{k}={points_finale[k]}\nG_{k}={gradients_finale[k]} \n \n")


