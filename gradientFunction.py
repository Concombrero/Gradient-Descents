from math import*
import numpy as np


def gradientFunction(function, evaluationPoint, step):
    """Calculate the gradient of a function at a given point

    Args:
        function (function): function whose gradient is calculated
        evaluationPoint (list): Coordinates of the evaluation point
        step (float): _description_
    
    Returns:
        list: gradient of the function at the given point
    """
    gradient=[]
    for i in range(len(evaluationPoint)):
        newPoint = np.copy(evaluationPoint)
        newPoint[i] += step
        rate=(function(newPoint)-function(evaluationPoint))/step
        gradient.append(rate)
    return np.array(gradient)

