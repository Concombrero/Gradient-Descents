from __future__ import annotations
from math import*
import numpy as np


def gradientFunction(function: callable, evaluationPoint: np.array , stepDerivative: float)->np.array:
    """Calculate the gradient of a function at a given point

    Args:
        function (function): function whose gradient is calculated
        evaluationPoint (array): coordinates of the evaluation point
        stepDerivative (float): size of the dx in the derivative approximation
    
    Returns:
        array: gradient of the function at the given point
    """
    gradient=[]
    for i in range(len(evaluationPoint)):
        newPoint = np.copy(evaluationPoint)
        newPoint[i] += stepDerivative
        partialDerivative=(function(newPoint)-function(evaluationPoint))/stepDerivative
        gradient.append(partialDerivative)
    return np.array(gradient)