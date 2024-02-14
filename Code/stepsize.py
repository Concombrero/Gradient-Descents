from gradientFunction import*

def backtracking(evaluationPoint : np.Array,gradient : np.array, stepDescent : float, function : function):
    """basic backtracking line search in the direction of the gradient

    Args:
        evaluationPoint (array): Initial Assessment Point
        gradient (array): gradient of the function at the given point
        stepDescent (float): initial step of the descent
        function (function): function from which we want to descend. The function must take an array in parameter even if it's à function with one variable.
    
    Returns : 
        float : step of the descent
    """
    while function(evaluationPoint+stepDescent*gradient)>function(evaluationPoint):
        stepDescent = stepDescent/2
    return stepDescent

def backtrackingDescent(function: function, evaluationPoint: np.array, stepDerivative:float, stepDescent:float, terminationCondition:float)->list[tuple[np.array, np.array]]:
    """Perform gradient descent on the specified function

    Args:
        function (function): function from which we want to descend. The function must take an array in parameter even if it's à function with one variable
        evaluationPoint (array): Initial Assessment Point
        stepDerivative (float): Deviation used to derive
        stepDescent (float): Initial Step of the descent
        terminationCondition (float): Value of the maximum gradient norm to stop the descent

    Returns:
        list[tupple(array,array)]: List of evaluation points through which the descent has passed and their assiociate gradient
    """
    path=[]
    gradient=gradientFunction(function, evaluationPoint, stepDerivative)
    path.append((evaluationPoint.copy(), gradient))
    while np.linalg.norm(gradient) >= terminationCondition:
        stepDescent = backtracking(evaluationPoint,gradient,stepDescent,function)
        evaluationPoint-=stepDescent*gradient
        gradient=gradientFunction(function, evaluationPoint, stepDerivative)
        path.append((evaluationPoint.copy(), gradient))
    return path