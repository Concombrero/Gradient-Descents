from gradientFunction import*

def momentumGradientDescent(function, evaluationPoint, stepDerivative, stepDescent, terminaisonCondition):
    path=[]
    t=[0]
    path.append(evaluationPoint.copy())
    gradient=gradientFunction(function, evaluationPoint, stepDerivative)
    evaluationPoint=-stepDescent*gradient
    path.append(evaluationPoint.copy)
    t.append((1/2)*(1+sqrt(1+4*t[-1]**2)))
    t.append((1/2)*(1+sqrt(1+4*t[-1]**2)))
    while np.linalg.norm(gradient)<=terminaisonCondition:
        t.append((1/2)*(1+sqrt(1+4*t[-1]**2)))
        B=(t[-2]-1)/(t[-1])
        gradient=gradientFunction(path[-1]+B*(path[-1]-path[-2]))
        evaluationPoint=(path[-1]-stepDescent*gradient+B*(path[-1]-path[-2]))
        path.append(evaluationPoint.copy())
    return path