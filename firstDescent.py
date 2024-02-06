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
        list[array]: List of evaluation points through which the descent has passed
    """
    path=[]
    path.append(evaluationPoint.copy())
    gradient=gradientFunction(function, evaluationPoint, stepDerivative)
    while np.linalg.norm(gradient) >= terminationCondition:
        evaluationPoint-=stepDescent*gradient
        path.append(evaluationPoint.copy())
        gradient=gradientFunction(function, evaluationPoint, stepDerivative)
    return path

