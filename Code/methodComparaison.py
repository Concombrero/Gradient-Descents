from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import random as rd
from descentAlgorythme import*

def createPoint(dimension: int, a:np.array, b:np.array)->np.array:
    point=[]
    for i in range(dimension):
        point.append(float(rd.randint(a[i],b[i])))
    return np.array(point)


def convergenceGraph(dimension: int, function: function, methods: list[function], nameMethode: list[str], a:np.array, b:np.array)->None:
    initialPoint=createPoint(dimension, a, b)
    for method in methods:
        Y=[]
        path=method(function, initialPoint.copy(), 0.01, 0.5, 1)
        for point in path:
            Y.append(np.linalg.norm(point[0]-path[-1][0]))
        X=[path[i][0] for i in range(len(path))]
        plt.plot(X,Y)
        plt.legend([nameMethode[i] for i in range(len(nameMethode))])
    plt.title('Convergence\'s rythm')
    plt.xlabel('Point')
    plt.ylabel('Distance to convergence point')
    plt.show()


def numberStepGraph(dimension: int, function: function, methods: list[function], nameMethode: list[str], precissons:list[float])->None:
    initialPoint=createPoint(dimension, 10, 15)
    precissions=[10., 1., 0.1, 0.01, 0.001]
    for method in methods:
        Y=[]
        for precission in precissions:
            path=method(function, initialPoint.copy(), 0.01, 0.1, precission)
            Y.append(len(path))
        plt.plot(precissions, Y)
    plt.legend([nameMethode[i] for i in range(len(nameMethode))])
    plt.semilogx()
    plt.title('Iteration in function of the value of the terminaison condition')
    plt.xlabel('Terminaison Condition')
    plt.ylabel('Number of step')
    plt.show()