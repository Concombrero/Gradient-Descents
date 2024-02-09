from gradientFunction import*

def firstDescent(function, evaluationPoint, stepDerivative, stepDescent, terminationCondition):
    """Perform gradient descent on the specified function

    Args:
        function (function): function from which we want to descend
        evaluationPoint (array): Initial Assessment Point
        stepDerivative (float): Deviation used to derive
        stepDescent (float): Steps of the descent
        terminationCondition (float): Value of the maximum gradient norm to stop the descent

    Returns:
        list[tupple(array,array)]: List of evaluation points through which the descent has passed and their assiociate gradient
    """
    path=[]
    gradient=gradientFunction(function, evaluationPoint, stepDerivative)
    path.append((evaluationPoint.copy(),gradient))
    while np.linalg.norm(gradient) >= terminationCondition:
        evaluationPoint-=stepDescent*gradient
        gradient=gradientFunction(function, evaluationPoint, stepDerivative)
        path.append((evaluationPoint.copy(),gradient))
    return path

def descentVarientStep(function, evaluationPoint, stepDerivative, stepDescent, terminationCondition):
    """Perform gradient descent on the specified function, calculating the new stepDescent at each iteration

    Args:
        function (function): function from which we want to descend
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

