from gradientFunction import*
import random

def stochasticGradient(function: callable, evaluationPoint: np.array , stepDerivative: float,sample_size : float)->np.array:
    """Calculate the gradient of a function at a given point

    Args:
        function (function): give a component of the function whose gradient is calculated, 
        with a default value of 0 corresponding to the function itself
        
        evaluationPoint (array): coordinates of the evaluation point
        
        stepDerivative (float): size of the dx in the derivative approximation
        
        sample_size (float) : number of f components (i.e. number of examples)
    
    Returns:
        array: gradient of one of the components of the function at the given point
    """
    gradient=[]
    for i in range(len(evaluationPoint)):
        newPoint = np.copy(evaluationPoint)
        newPoint[i] += stepDerivative
        sample = random.randint(1, sample_size)
        partialDerivative=(function(newPoint,sample)-function(evaluationPoint,sample))/stepDerivative
        gradient.append(partialDerivative)
    return np.array(gradient)

# peut-être réunir avec la descente classique, avec la valeur par défaut pour les paramètres supplémentaires

def lotGradient(function: callable, evaluationPoint: np.array , stepDerivative: float,batchsize : float, sample_size : float)->np.array:
    """Calculate the gradient of a function at a given point

    Args:
        function (function): give a component of the function whose gradient is calculated, 
        with a default value of 0 corresponding to the function itself
        
        evaluationPoint (array): coordinates of the evaluation point
        
        stepDerivative (float): size of the dx in the derivative approximation
        
        batchsize (float) : number of f components taken into account for the gradient
        
        sample_size (float) : number of f components (i.e. number of examples)
        
    
    Returns:
        array: gradient of one of the components of the function at the given point
    """
    gradient=[]
    for i in range(len(evaluationPoint)):
        newPoint = np.copy(evaluationPoint)
        newPoint[i] += stepDerivative
        sample = [random.randint(1, sample_size) for j in range(batchsize)]
        partialDerivative=(sum(function(newPoint,g) for g in sample)-sum(function(evaluationPoint,g) for g in sample))/stepDerivative
        gradient.append(partialDerivative)
    return np.array(gradient)
