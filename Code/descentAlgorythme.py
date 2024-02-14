from __future__ import annotations
from gradientFunction import*

def firstDescent(function: function, evaluationPoint: np.array, stepDerivative:float, stepDescent:float, terminationCondition:float)->list[tuple[np.array, np.array]]:
    """Perform gradient descent on the specified function

    Args:
        function (function): function from which we want to descend. The function must take an array in parameter even if it's à function with one variable
        evaluationPoint (array): Initial Assessment Point
        stepDerivative (float): Deviation used to derive
        stepDescent (float): Steps of the descent
        terminationCondition (float): Value of the maximum gradient norm to stop the descent

    Returns:
        list[tupple(array,array)]: List of evaluation points through which the descent has passed and their assiociate gradient
    """
    path=[]
    gradient=gradientFunction(function, evaluationPoint, stepDerivative)
    path.append((evaluationPoint.copy(), gradient))
    while np.linalg.norm(gradient) >= terminationCondition:
        evaluationPoint-=stepDescent*gradient
        gradient=gradientFunction(function, evaluationPoint, stepDerivative)
        path.append((evaluationPoint.copy(), gradient))
    return path

def descentVarientStep(function:function, evaluationPoint:np.array, stepDerivative: float, stepDescent:function, terminationCondition:float)->list[tuple[np.array, np.array]]:
    """Perform gradient descent on the specified function, calculating the new stepDescent at each iteration

    Args:
        function (function): function from which we want to descend. The function must take an array in parameter even if it's à function with one variable.
        evaluationPoint (np.array): Initial Assessment Point
        stepDerivative (float): Deviation used to derive
        stepDescent (function): Function calculating the kth step of descent
        terminationCondition (float): Value of the maximum gradient norm to stop the descent
        
    Returns:
        list[tupple[np.array,np.array]]: List of evaluation points through which the descent has passed and their assiociate gradient
    """
    path=[]
    counterIteration=0
    gradient=gradientFunction(function, evaluationPoint, stepDerivative)
    path.append(evaluationPoint.copy(),gradient)
    while np.linalg.norm(gradient) >= terminationCondition:
        evaluationPoint-=stepDescent(counterIteration)*gradient
        gradient=gradientFunction(function, evaluationPoint, stepDerivative)
        path.append((evaluationPoint.copy(),gradient))
        counterIteration+=1
    return path

def heavyBallDescent(function:function, evaluationPoint:np.array, stepDerivative:float, stepDescent:float, terminaisonCondition:float, stepMomentum: float)->list[tuple[np.array, np.array]]:
    """Perform the heavy ball gradient descent on the specified function

    Args:
        function (function): function from which we want to descend. The function must take an array in parameter even if it's à function with one variable.
        evaluationPoint (np.array): Initial Assessment Point
        stepDerivative (float): Deviation used to derive
        stepDescent (float): Steps of the descent
        terminationCondition (float): Value of the maximum momentum norm to stop the descent
        stepMomentum (float): A value between 0 and 1 indicating the importance of the previous term for the recalculation of momentum

    Returns:
        list[tupple[np.array,np.array]]: List of evaluation points through which the descent has passed and their assiociate gradient
    """
    path=[]
    momentum=gradientFunction(function, evaluationPoint, stepDerivative)
    path.append((evaluationPoint.copy(), momentum.copy()))
    while np.linalg.norm(gradientFunction(function, evaluationPoint, stepDerivative)) >= terminaisonCondition:
        evaluationPoint-=stepDescent*momentum
        momentum=stepMomentum*momentum+(1-stepMomentum)*gradientFunction(function, evaluationPoint, stepDerivative)
        path.append((evaluationPoint.copy(), momentum.copy()))
    return path

def nesterovDescent(function: function, evaluationPoint: np.array, stepDerivative: float, stepDescent:float, terminaisonCondition:float)->list[tuple[np.array, np.array]]:
    """Perform the heavy ball gradient descent on the specified function

    Args:
        function (function): function from which we want to descend. The function must take an array in parameter even if it's à function with one variable.
        evaluationPoint (np.array): Initial Assessment Point
        stepDerivative (float): Deviation used to derive
        stepDescent (float): Steps of the descent
        terminationCondition (float): Value of the maximum momentum norm to stop the descent

    Returns:
        list[tupple[np.array,np.array]]: List of evaluation points through which the descent has passed and their assiociate gradient
    """
    path=[]
    lambdaPrevius=0
    lambdaCurrent=1
    beta=1
    path.append(evaluationPoint.copy())
    path.append(evaluationPoint.copy())
    while np.linalg.norm(gradientFunction(function, evaluationPoint, stepDerivative)) >= terminaisonCondition:
        evaluationPoint=evaluationPoint-stepDerivative*gradientFunction(function, path[-1]+beta*(path[-1]-path[-2]), stepDerivative)+beta*(path[-1]-path[-2])
        
        lambdaTemp=lambdaCurrent
        lambdaCurrent=(1+sqrt(1+4*lambdaCurrent**2))/2
        lambdaPrevius=lambdaTemp
        
        beta=(lambdaPrevius-1)/lambdaCurrent
        
        path.append(evaluationPoint)
    return path[1::]

