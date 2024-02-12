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
        evaluationPoint (array): Initial Assessment Point
        stepDerivative (float): Deviation used to derive
        stepDescent (float): Steps of the descent
        terminationCondition (float): Value of the maximum gradient norm to stop the descent
        
    Returns:
        list[tupple(array,array)]: List of evaluation points through which the descent has passed and their assiociate gradient
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

