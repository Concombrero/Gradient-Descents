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

def heavyBallDescent(function:function, evaluationPoint:np.array, stepDerivative:float, stepDescent:float, terminationCondition:float, stepMomentum: float =0.9, momentumTermination:float =0.5)->list[tuple[np.array, np.array]]:
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
    while np.linalg.norm(gradientFunction(function, evaluationPoint, stepDerivative)) >= terminationCondition or np.linalg.norm(momentum)>momentumTermination:
        evaluationPoint-=stepDescent*momentum
        momentum=stepMomentum*momentum+(1-stepMomentum)*gradientFunction(function, evaluationPoint, stepDerivative)
        path.append((evaluationPoint.copy(), momentum.copy()))
    return path

def nesterovDescent(function: function, evaluationPoint: np.array, stepDerivative: float, stepDescent:float, terminationCondition:float, momentumTermination:float =0.5)->list[tuple[np.array, np.array]]:
    """Perform the nesterov descent gradient descent on the specified function

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
    lambdaPrevious=0
    lambdaCurrent=1
    beta=1
    path.append(evaluationPoint.copy())
    path.append(evaluationPoint.copy())
    while np.linalg.norm(gradientFunction(function, evaluationPoint, stepDerivative)) >= terminationCondition or np.linalg.norm(beta*(path[-1]-path[-2]))>momentumTermination:
        evaluationPoint=evaluationPoint-stepDerivative*gradientFunction(function, path[-1]+beta*(path[-1]-path[-2]), stepDerivative)+beta*(path[-1]-path[-2])
        
        lambdaTemp=lambdaCurrent
        lambdaCurrent=(1+sqrt(1+4*lambdaCurrent**2))/2
        lambdaPrevious=lambdaTemp
        
        beta=(lambdaPrevious-1)/lambdaCurrent
        
        path.append(evaluationPoint)
    return path[1::]

def adamDescent(function, evaluationPoint, stepDerivative, stepDescent, terminationCondition, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Perform gradient descent using Adam optimizer on the specified function

    Args:
        function (function): function from which we want to descend. The function must take an array in parameter even if it's a function with one variable.
        evaluationPoint (array): Initial assessment point.
        stepDerivative (float): Deviation used to derive.
        stepDescent (float): Steps of the descent.
        terminationCondition (float): Value of the maximum gradient norm to stop the descent.
        learning_rate (float): Learning rate for the optimizer.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        list[tuple(array,array)]: List of evaluation points through which the descent has passed and their associated gradient.
    """
    path=[]
    m = np.zeros_like(evaluationPoint)
    v = np.zeros_like(evaluationPoint)
    beta1_t = 1.0
    beta2_t = 1.0
    path.append((evaluationPoint.copy(), gradientFunction(function, evaluationPoint, stepDerivative)))
    while np.linalg.norm(gradientFunction(function, evaluationPoint, stepDerivative)) >= terminationCondition:
        g = gradientFunction(function, evaluationPoint, stepDerivative)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1_t)
        v_hat = v / (1 - beta2_t)
        evaluationPoint -= stepDescent * learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        beta1_t *= beta1
        beta2_t *= beta2
        path.append((evaluationPoint.copy(), gradientFunction(function, evaluationPoint, stepDerivative)))
    return path

def rmspropDescent(function, evaluationPoint, stepDerivative, stepDescent, terminationCondition, learning_rate=0.001, beta=0.9, epsilon=1e-8):
    """Perform gradient descent using RMSprop optimizer on the specified function

    Args:
        function (function): function from which we want to descend. The function must take an array in parameter even if it's a function with one variable.
        evaluationPoint (array): Initial assessment point.
        stepDerivative (float): Deviation used to derive.
        stepDescent (float): Steps of the descent.
        terminationCondition (float): Value of the maximum gradient norm to stop the descent.
        learning_rate (float): Learning rate for the optimizer.
        beta (float): Exponential decay rate for the estimation of squared gradients.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        list[tuple(array,array)]: List of evaluation points through which the descent has passed and their associated gradient.
    """
    path=[]
    v = np.zeros_like(evaluationPoint)
    path.append((evaluationPoint.copy(), gradientFunction(function, evaluationPoint, stepDerivative)))
    while np.linalg.norm(gradientFunction(function, evaluationPoint, stepDerivative)) >= terminationCondition:
        g = gradientFunction(function, evaluationPoint, stepDerivative)
        v = beta * v + (1 - beta) * g**2
        evaluationPoint -= stepDescent * learning_rate * g / (np.sqrt(v) + epsilon)
        path.append((evaluationPoint.copy(), gradientFunction(function, evaluationPoint, stepDerivative)))
    return path

def adagradDescent(function, evaluationPoint, stepDerivative, stepDescent, terminationCondition, learning_rate=0.001, epsilon=1e-8):
    """Perform gradient descent using Adagrad optimizer on the specified function

    Args:
        function (function): function from which we want to descend. The function must take an array in parameter even if it's a function with one variable.
        evaluationPoint (array): Initial assessment point.
        stepDerivative (float): Deviation used to derive.
        stepDescent (float): Steps of the descent.
        terminationCondition (float): Value of the maximum gradient norm to stop the descent.
        learning_rate (float): Learning rate for the optimizer.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        list[tuple(array,array)]: List of evaluation points through which the descent has passed and their associated gradient.
    """
    path = []
    v = np.zeros_like(evaluationPoint)
    path.append((evaluationPoint.copy(), gradientFunction(function, evaluationPoint, stepDerivative)))
    while np.linalg.norm(gradientFunction(function, evaluationPoint, stepDerivative)) >= terminationCondition:
        g = gradientFunction(function, evaluationPoint, stepDerivative)
        v += g**2
        evaluationPoint -= stepDescent * learning_rate * g / (np.sqrt(v) + epsilon)
        path.append((evaluationPoint.copy(), gradientFunction(function, evaluationPoint, stepDerivative)))
    return path